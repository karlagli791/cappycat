//! Job manager shared by the pipeline runner, the voice separator and the exporter.
//!
//! Every job — a spawned process ([`JobManager::spawn`]) or an in-process task
//! that drives several children itself (the compositor export) — is a
//! [`TaskCtl`] addressed by its id, so the frontend can cancel it. Events are
//! pushed through the [`EventSink`] abstraction (implemented for
//! `tauri::AppHandle`; tests use [`CollectingSink`]).
//!
//! **One GPU job at a time**: analysis (`run_pipeline`), `separate_audio` and
//! the optical-flow `interpolate` runs of exports share the FIFO
//! [`GpuQueue`]. A job that has to wait emits its normal progress event with
//! [`GPU_WAIT_MESSAGE`] (pct 0) and starts when every job queued before it has
//! finished; cancelling a queued job removes it from the queue.
//!
//! Cancelling never blocks the caller: ffmpeg children are killed at once,
//! python children first get their stdin closed (the pipeline's watchdog then
//! exits and stops its own ffmpeg children) and are killed — with their whole
//! process tree — after [`crate::procs::PYTHON_GRACE`].

use crate::procs::{self, ManagedChild, TreeJob};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Progress message of a job waiting for the GPU queue.
pub const GPU_WAIT_MESSAGE: &str = "waiting for the GPU (another AI job is running)";

/// Something that can deliver a named event with a JSON payload to the UI.
pub trait EventSink: Send + Sync + 'static {
    fn emit(&self, event: &str, payload: Value);
}

impl EventSink for tauri::AppHandle {
    fn emit(&self, event: &str, payload: Value) {
        if let Err(e) = tauri::Emitter::emit(self, event, payload) {
            tracing::warn!("failed to emit {event}: {e}");
        }
    }
}

/// In-memory sink used by unit tests.
#[derive(Default)]
pub struct CollectingSink {
    pub events: Mutex<Vec<(String, Value)>>,
}

impl CollectingSink {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
    pub fn snapshot(&self) -> Vec<(String, Value)> {
        self.events.lock().unwrap().clone()
    }
    pub fn named(&self, name: &str) -> Vec<Value> {
        self.snapshot().into_iter().filter(|(n, _)| n == name).map(|(_, v)| v).collect()
    }
}

impl EventSink for CollectingSink {
    fn emit(&self, event: &str, payload: Value) {
        self.events.lock().unwrap().push((event.to_string(), payload));
    }
}

/// Callbacks for one job's process I/O.
pub struct JobIo {
    pub on_stdout: Box<dyn FnMut(&str) + Send>,
    pub on_stderr: Box<dyn FnMut(&str) + Send>,
    /// Called once with the exit status (`None` when the process never ran or
    /// could not be waited for) and whether the job was cancelled.
    pub on_exit: Box<dyn FnOnce(Option<ExitStatus>, bool) + Send>,
}

/// How [`JobManager::spawn`] runs a command.
#[derive(Default)]
pub struct SpawnOpts {
    /// A `python -m cappycat_pipeline …` child: stdin is piped and held open
    /// (`CAPPYCAT_WATCH_STDIN=1`), the process tree gets its own nested job.
    pub python: bool,
    /// Run through the GPU queue. While the job waits because another GPU job
    /// is running, the callback is invoked shortly after `spawn` returns (so the
    /// caller already knows the job id) and then every 2 s until it starts.
    pub gpu_wait: Option<Box<dyn Fn() + Send + Sync>>,
}

/* ------------------------------------------------------------------ TaskCtl */

/// Control block of one job. Cancelling sets the flag and stops every
/// registered child so blocking reads return promptly.
#[derive(Default)]
pub struct TaskCtl {
    cancelled: AtomicBool,
    done: AtomicBool,
    children: Mutex<Vec<Arc<ManagedChild>>>,
}

impl TaskCtl {
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    fn track(&self, m: Arc<ManagedChild>) {
        {
            let mut list = self.children.lock().unwrap();
            // Forget children that have already exited.
            list.retain(|c| !c.has_exited());
            list.push(m.clone());
        }
        if self.is_cancelled() {
            stop_async(m);
        }
    }

    /// Track an ffmpeg-style child so [`TaskCtl::cancel`] can kill it. If the
    /// task is already cancelled the child is killed immediately.
    pub fn register(&self, child: Child) -> Arc<Mutex<Child>> {
        let m = Arc::new(ManagedChild { child: Arc::new(Mutex::new(child)), stdin: Mutex::new(None), tree: None });
        let c = m.child.clone();
        self.track(m);
        c
    }

    /// Track a python child: its stdin is held open until the child should go,
    /// and its process tree is killed together on cancel.
    pub fn register_python(&self, child: Child, stdin: Option<ChildStdin>) -> Arc<ManagedChild> {
        let tree = TreeJob::for_child(&child);
        let m = Arc::new(ManagedChild { child: Arc::new(Mutex::new(child)), stdin: Mutex::new(stdin), tree: Some(tree) });
        self.track(m.clone());
        m
    }

    /// Cancel: never blocks (python children are stopped on background threads).
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        let list: Vec<Arc<ManagedChild>> = self.children.lock().unwrap().clone();
        for c in list {
            stop_async(c);
        }
    }

    pub fn mark_done(&self) {
        self.done.store(true, Ordering::SeqCst);
    }

    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::SeqCst)
    }
}

fn stop_async(c: Arc<ManagedChild>) {
    if c.is_python() {
        let _ = std::thread::Builder::new().name("job-stop".into()).spawn(move || c.stop());
    } else {
        c.stop();
    }
}

/* ----------------------------------------------------------------- GPU queue */

/// FIFO queue that lets one GPU job run at a time.
#[derive(Default)]
pub struct GpuQueue {
    state: Mutex<QueueState>,
    cv: Condvar,
}

#[derive(Default)]
struct QueueState {
    running: Option<String>,
    waiting: VecDeque<String>,
}

/// Held while a job owns the GPU; dropping it starts the next queued job.
pub struct GpuGuard {
    queue: Arc<GpuQueue>,
    id: String,
}

impl Drop for GpuGuard {
    fn drop(&mut self) {
        let mut st = self.queue.state.lock().unwrap();
        if st.running.as_deref() == Some(self.id.as_str()) {
            st.running = None;
        }
        drop(st);
        self.queue.cv.notify_all();
    }
}

impl GpuQueue {
    /// Take the GPU now if it is free and nobody is queued.
    pub fn try_acquire(self: &Arc<Self>, id: &str) -> Option<GpuGuard> {
        let mut st = self.state.lock().unwrap();
        if st.running.is_none() && st.waiting.is_empty() {
            st.running = Some(id.to_string());
            return Some(GpuGuard { queue: self.clone(), id: id.to_string() });
        }
        None
    }

    /// Take the GPU if it is free, else join the end of the queue (atomically).
    pub fn acquire_or_enqueue(self: &Arc<Self>, id: &str) -> Option<GpuGuard> {
        let mut st = self.state.lock().unwrap();
        if st.running.is_none() && st.waiting.is_empty() {
            st.running = Some(id.to_string());
            return Some(GpuGuard { queue: self.clone(), id: id.to_string() });
        }
        st.waiting.push_back(id.to_string());
        None
    }

    /// Wait (FIFO) for the GPU. `on_wait` runs once if the job has to queue.
    /// Returns `None` when `ctl` is cancelled while queued (the job leaves the queue).
    pub fn acquire(self: &Arc<Self>, id: &str, ctl: &TaskCtl, on_wait: impl FnOnce()) -> Option<GpuGuard> {
        if let Some(g) = self.acquire_or_enqueue(id) {
            return Some(g);
        }
        on_wait();
        self.wait_turn(id, ctl)
    }

    /// Wait until `id` (already queued by [`GpuQueue::acquire_or_enqueue`]) is at the
    /// front and the GPU is free. `None` when cancelled while waiting.
    pub fn wait_turn(self: &Arc<Self>, id: &str, ctl: &TaskCtl) -> Option<GpuGuard> {
        self.wait_turn_with(id, ctl, &mut || {}, Duration::MAX, Duration::MAX)
    }

    /// [`GpuQueue::wait_turn`] calling `heartbeat` after `first` and then every `every`
    /// while still waiting.
    pub fn wait_turn_with(self: &Arc<Self>, id: &str, ctl: &TaskCtl, heartbeat: &mut dyn FnMut(), first: Duration, every: Duration) -> Option<GpuGuard> {
        let start = Instant::now();
        let mut next_beat = first;
        let mut st = self.state.lock().unwrap();
        loop {
            if start.elapsed() >= next_beat {
                drop(st);
                heartbeat();
                next_beat = next_beat.saturating_add(every);
                st = self.state.lock().unwrap();
            }
            if ctl.is_cancelled() {
                st.waiting.retain(|w| w != id);
                drop(st);
                self.cv.notify_all();
                return None;
            }
            if st.running.is_none() && st.waiting.front().map(String::as_str) == Some(id) {
                st.waiting.pop_front();
                st.running = Some(id.to_string());
                return Some(GpuGuard { queue: self.clone(), id: id.to_string() });
            }
            st = self.cv.wait_timeout(st, Duration::from_millis(100)).unwrap().0;
        }
    }

    /// Wake waiters (after a cancel) so they notice promptly.
    pub fn notify(&self) {
        self.cv.notify_all();
    }

    /// `(running, waiting in order)` — for tests and diagnostics.
    pub fn snapshot(&self) -> (Option<String>, Vec<String>) {
        let st = self.state.lock().unwrap();
        (st.running.clone(), st.waiting.iter().cloned().collect())
    }
}

/* ---------------------------------------------------------------- JobManager */

#[derive(Default)]
pub struct JobManager {
    tasks: Mutex<HashMap<String, Arc<TaskCtl>>>,
    /// The GPU queue shared by analysis, voice separation and optical flow.
    pub gpu: Arc<GpuQueue>,
}

impl JobManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_job_id(prefix: &str) -> String {
        format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
    }

    pub fn is_running(&self, job_id: &str) -> bool {
        self.tasks.lock().unwrap().get(job_id).map(|t| !t.is_done()).unwrap_or(false)
    }

    pub fn running_ids(&self) -> Vec<String> {
        self.tasks.lock().unwrap().keys().cloned().collect()
    }

    /// Register a job; cancel it with [`JobManager::cancel`] and call
    /// [`TaskCtl::mark_done`] + [`JobManager::finish`] when it ends.
    pub fn register_task(&self, job_id: &str) -> Arc<TaskCtl> {
        let ctl = Arc::new(TaskCtl::default());
        self.tasks.lock().unwrap().insert(job_id.to_string(), ctl.clone());
        ctl
    }

    /// Spawn `cmd` under `job_id`, wiring stdout/stderr to the callbacks.
    /// Errors to start the process are returned synchronously unless the job
    /// had to queue for the GPU (then they arrive through `on_stderr` + `on_exit`).
    pub fn spawn(&self, job_id: &str, mut cmd: Command, io: JobIo, opts: SpawnOpts) -> Result<(), String> {
        let ctl = self.register_task(job_id);
        let SpawnOpts { python, gpu_wait } = opts;
        let Some(on_wait) = gpu_wait else {
            return launch(job_id, &ctl, &mut cmd, io, python, None).map_err(|(e, _)| {
                self.finish(job_id);
                e
            });
        };
        if let Some(guard) = self.gpu.acquire_or_enqueue(job_id) {
            return launch(job_id, &ctl, &mut cmd, io, python, Some(guard)).map_err(|(e, _)| {
                self.finish(job_id);
                e
            });
        }
        let gpu = self.gpu.clone();
        let id = job_id.to_string();
        std::thread::Builder::new()
            .name(format!("{job_id}-queued"))
            .spawn(move || match gpu.wait_turn_with(&id, &ctl, &mut || on_wait(), Duration::from_millis(150), Duration::from_secs(2)) {
                None => {
                    (io.on_exit)(None, true);
                    ctl.mark_done();
                }
                Some(guard) => {
                    if let Err((e, mut io)) = launch(&id, &ctl, &mut cmd, io, python, Some(guard)) {
                        (io.on_stderr)(&e);
                        (io.on_exit)(None, ctl.is_cancelled());
                        ctl.mark_done();
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Cancel a job (queued or running). Never blocks.
    pub fn cancel(&self, job_id: &str) -> Result<(), String> {
        let t = self.tasks.lock().unwrap().get(job_id).cloned();
        match t {
            Some(t) => {
                t.cancel();
                self.gpu.notify();
                Ok(())
            }
            None => Err(format!("unknown job {job_id}")),
        }
    }

    /// Cancel every job (app shutdown).
    pub fn cancel_all(&self) {
        let all: Vec<Arc<TaskCtl>> = self.tasks.lock().unwrap().values().cloned().collect();
        for t in all {
            t.cancel();
        }
        self.gpu.notify();
    }

    /// Drop the bookkeeping entry for a finished job.
    pub fn finish(&self, job_id: &str) {
        self.tasks.lock().unwrap().remove(job_id);
    }

    /// Block until the job has exited (test helper). Returns false on timeout.
    pub fn wait(&self, job_id: &str, timeout: Duration) -> bool {
        let start = Instant::now();
        loop {
            let done = self.tasks.lock().unwrap().get(job_id).map(|t| t.is_done()).unwrap_or(true);
            if done {
                return true;
            }
            if start.elapsed() > timeout {
                return false;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

/// Start the process and its reader / waiter threads. On failure the
/// callbacks are handed back so the caller can report the error.
fn launch(
    job_id: &str,
    ctl: &Arc<TaskCtl>,
    cmd: &mut Command,
    io: JobIo,
    python: bool,
    gpu: Option<GpuGuard>,
) -> Result<(), (String, JobIo)> {
    if ctl.is_cancelled() {
        return Err(("cancelled".into(), io));
    }
    if python {
        cmd.env("CAPPYCAT_WATCH_STDIN", "1").env("PYTHONUNBUFFERED", "1").env("PYTHONIOENCODING", "utf-8");
    }
    cmd.stdin(if python { Stdio::piped() } else { Stdio::null() }).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = match procs::spawn(cmd) {
        Ok(c) => c,
        Err(e) => return Err((format!("failed to spawn {:?}: {e}", cmd.get_program()), io)),
    };
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdin = child.stdin.take();
    let managed: Arc<ManagedChild> = if python {
        ctl.register_python(child, stdin)
    } else {
        let c = ctl.register(child);
        Arc::new(ManagedChild { child: c, stdin: Mutex::new(None), tree: None })
    };

    let JobIo { mut on_stdout, mut on_stderr, on_exit } = io;
    let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
    let tx2 = done_tx.clone();
    let spawn_reader = |name: String, f: Box<dyn FnOnce() + Send>| std::thread::Builder::new().name(name).spawn(f);
    let r1 = spawn_reader(
        format!("{job_id}-stdout"),
        Box::new(move || {
            if let Some(s) = stdout {
                for line in BufReader::new(s).lines().map_while(Result::ok) {
                    on_stdout(&line);
                }
            }
            let _ = done_tx.send(());
        }),
    );
    let r2 = spawn_reader(
        format!("{job_id}-stderr"),
        Box::new(move || {
            if let Some(s) = stderr {
                for line in BufReader::new(s).lines().map_while(Result::ok) {
                    on_stderr(&line);
                }
            }
            let _ = tx2.send(());
        }),
    );
    if r1.is_err() || r2.is_err() {
        ctl.cancel();
    }
    let id = job_id.to_string();
    let ctl2 = ctl.clone();
    let waiter = std::thread::Builder::new().name(format!("{job_id}-wait")).spawn(move || {
        let status = procs::wait_child(&managed.child).ok();
        // Let the readers deliver the last lines (a result line before the exit
        // event), but don't hang on pipes a surviving grandchild still holds.
        let deadline = Instant::now() + Duration::from_secs(5);
        for _ in 0..2 {
            let left = deadline.saturating_duration_since(Instant::now());
            if done_rx.recv_timeout(left).is_err() {
                break;
            }
        }
        if let Some(t) = &managed.tree {
            t.kill(); // no orphans once the job is over
        }
        drop(gpu);
        // on_exit runs before `done` flips so `wait()` observers see its side effects.
        on_exit(status, ctl2.is_cancelled());
        ctl2.mark_done();
        tracing::debug!("job {id} finished");
    });
    if let Err(e) = waiter {
        ctl.cancel();
        tracing::error!("cannot start the waiter thread of {job_id}: {e}");
    }
    Ok(())
}

/// Locate a `python` interpreter: `CAPPYCAT_PYTHON` (a file), else
/// `<pipeline_dir>/.venv/Scripts/python.exe` (or `bin/python` on unix) if present, else the
/// managed env `setup_ai` creates (`%LOCALAPPDATA%\Cappycat\python`), else the first `python` on
/// PATH.
pub fn find_python(pipeline_dir: Option<&std::path::Path>) -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("CAPPYCAT_PYTHON").map(std::path::PathBuf::from) {
        if p.is_file() {
            return Some(p);
        }
    }
    let managed = crate::paths::resolver().managed_python();
    // installed: the managed env first (a stray `.venv` next to the bundled pipeline is not ours)
    if crate::paths::is_installed() && managed.is_file() {
        return Some(managed);
    }
    if let Some(dir) = pipeline_dir {
        #[cfg(windows)]
        let venv = dir.join(".venv").join("Scripts").join("python.exe");
        #[cfg(not(windows))]
        let venv = dir.join(".venv").join("bin").join("python");
        if venv.is_file() {
            return Some(venv);
        }
    }
    if managed.is_file() {
        return Some(managed);
    }
    let names: &[&str] = if cfg!(windows) { &["python.exe", "python3.exe"] } else { &["python3", "python"] };
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for n in names {
            let p = dir.join(n);
            if p.is_file() {
                // Skip the Microsoft Store alias stub, which just opens the Store.
                if p.to_string_lossy().contains("WindowsApps") {
                    continue;
                }
                return Some(p);
            }
        }
    }
    None
}

/// `python -m cappycat_pipeline …` command skeleton: cwd = pipeline dir, unbuffered UTF-8
/// output and the stdin watchdog switched on (`CAPPYCAT_WATCH_STDIN=1`; the caller must pipe
/// stdin and keep the handle until the child should stop).
pub fn python_command(python: &std::path::Path, pipeline_dir: &std::path::Path) -> Command {
    let mut cmd = Command::new(python);
    cmd.current_dir(pipeline_dir)
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("CAPPYCAT_WATCH_STDIN", "1");
    // models / characters / stems / ffmpeg locations of the current layout (crate::paths)
    for (k, v) in crate::paths::python_env() {
        cmd.env(k, v);
    }
    cmd
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn python_or_skip() -> Option<std::path::PathBuf> {
        let p = find_python(None);
        if p.is_none() {
            eprintln!("SKIP: python not on PATH");
        }
        p
    }

    #[test]
    fn spawn_reads_lines_and_reports_exit() {
        let Some(python) = python_or_skip() else { return };
        let jobs = Arc::new(JobManager::new());
        let sink = CollectingSink::new();
        let mut cmd = Command::new(python);
        cmd.args(["-c", "import sys; print('hello'); print('world'); print('warn', file=sys.stderr); sys.exit(3)"]);
        let (s1, s2, s3) = (sink.clone(), sink.clone(), sink.clone());
        let j = jobs.clone();
        jobs.spawn(
            "job_test",
            cmd,
            JobIo {
                on_stdout: Box::new(move |l| s1.emit("out", json!(l))),
                on_stderr: Box::new(move |l| s2.emit("err", json!(l))),
                on_exit: Box::new(move |status, cancelled| {
                    s3.emit("exit", json!({ "code": status.and_then(|s| s.code()), "cancelled": cancelled }));
                    j.finish("job_test");
                }),
            },
            SpawnOpts::default(),
        )
        .unwrap();
        assert!(jobs.wait("job_test", Duration::from_secs(30)));
        std::thread::sleep(Duration::from_millis(50));
        let out: Vec<Value> = sink.named("out");
        assert_eq!(out, vec![json!("hello"), json!("world")]);
        assert_eq!(sink.named("err"), vec![json!("warn")]);
        let exit = &sink.named("exit")[0];
        assert_eq!(exit["code"], 3);
        assert_eq!(exit["cancelled"], false);
        assert!(!jobs.is_running("job_test"));
    }

    #[test]
    fn cancel_kills_process() {
        let Some(python) = python_or_skip() else { return };
        let jobs = Arc::new(JobManager::new());
        let sink = CollectingSink::new();
        let mut cmd = Command::new(python);
        cmd.args(["-c", "import time; print('start', flush=True); time.sleep(30)"]);
        let s = sink.clone();
        let j = jobs.clone();
        jobs.spawn(
            "job_cancel",
            cmd,
            JobIo {
                on_stdout: Box::new(|_| {}),
                on_stderr: Box::new(|_| {}),
                on_exit: Box::new(move |_, cancelled| {
                    s.emit("exit", json!({ "cancelled": cancelled }));
                    j.finish("job_cancel");
                }),
            },
            SpawnOpts::default(),
        )
        .unwrap();
        assert!(jobs.is_running("job_cancel"));
        let t = Instant::now();
        jobs.cancel("job_cancel").unwrap();
        assert!(t.elapsed() < Duration::from_millis(500), "cancel must not block");
        assert!(jobs.wait("job_cancel", Duration::from_secs(10)));
        assert!(t.elapsed() < Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(sink.named("exit")[0]["cancelled"], true);
        assert!(jobs.cancel("job_cancel").is_err(), "finished job is forgotten");
    }

    /// A python child ignoring stdin (no watchdog) is killed after the grace period;
    /// one that honours stdin EOF exits on its own right away.
    #[test]
    fn python_children_get_stdin_eof_then_a_kill() {
        let Some(python) = python_or_skip() else { return };
        for (script, max_secs) in [
            ("import sys, time; print('up', flush=True); sys.stdin.read(); print('eof', flush=True)", 1.5),
            ("import time; print('up', flush=True); time.sleep(60)", 6.0),
        ] {
            let jobs = Arc::new(JobManager::new());
            let sink = CollectingSink::new();
            let mut cmd = Command::new(&python);
            cmd.args(["-c", script]);
            let (s1, s2) = (sink.clone(), sink.clone());
            jobs.spawn(
                "job_py",
                cmd,
                JobIo {
                    on_stdout: Box::new(move |l| s1.emit("out", json!(l))),
                    on_stderr: Box::new(|_| {}),
                    on_exit: Box::new(move |_, c| s2.emit("exit", json!({ "cancelled": c }))),
                },
                SpawnOpts { python: true, gpu_wait: None },
            )
            .unwrap();
            let t0 = Instant::now();
            while sink.named("out").is_empty() && t0.elapsed() < Duration::from_secs(20) {
                std::thread::sleep(Duration::from_millis(20));
            }
            let t = Instant::now();
            jobs.cancel("job_py").unwrap();
            assert!(jobs.wait("job_py", Duration::from_secs(15)));
            let took = t.elapsed().as_secs_f64();
            assert!(took < max_secs, "{script}: stopped after {took:.2}s");
            assert_eq!(sink.named("exit")[0]["cancelled"], true);
            if script.contains("stdin.read") {
                assert!(sink.named("out").iter().any(|l| l == "eof"), "stdin EOF reached the child: {:?}", sink.named("out"));
            }
        }
    }

    #[test]
    fn gpu_queue_is_fifo_and_cancel_removes_queued_jobs() {
        let q = Arc::new(GpuQueue::default());
        let order = Arc::new(Mutex::new(Vec::<String>::new()));
        let first = q.try_acquire("a").expect("free queue");
        assert!(q.try_acquire("x").is_none(), "busy");
        let ctls: Vec<Arc<TaskCtl>> = (0..3).map(|_| Arc::new(TaskCtl::default())).collect();
        let waits = Arc::new(Mutex::new(Vec::<String>::new()));
        let mut handles = Vec::new();
        for (i, name) in ["b", "c", "d"].iter().enumerate() {
            let (q2, order, ctl, waits) = (q.clone(), order.clone(), ctls[i].clone(), waits.clone());
            let name = name.to_string();
            handles.push(std::thread::spawn(move || {
                let q = q2;
                let n2 = name.clone();
                let g = q.acquire(&name, &ctl, || waits.lock().unwrap().push(n2));
                if let Some(g) = g {
                    order.lock().unwrap().push(name);
                    std::thread::sleep(Duration::from_millis(30));
                    drop(g);
                }
            }));
            // enqueue deterministically in order
            let t0 = Instant::now();
            while q.snapshot().1.len() < i + 1 && t0.elapsed() < Duration::from_secs(5) {
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        assert_eq!(q.snapshot(), (Some("a".into()), vec!["b".into(), "c".into(), "d".into()]));
        assert_eq!(*waits.lock().unwrap(), vec!["b", "c", "d"], "every queued job was told it waits");
        // cancel "c" while queued: it leaves the queue
        ctls[1].cancel();
        q.notify();
        let t0 = Instant::now();
        while q.snapshot().1.len() != 2 && t0.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(q.snapshot().1, vec!["b".to_string(), "d".to_string()]);
        drop(first);
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(*order.lock().unwrap(), vec!["b", "d"], "FIFO order, cancelled job skipped");
        assert_eq!(q.snapshot(), (None, vec![]));
    }

    #[test]
    fn queued_process_jobs_wait_for_the_gpu() {
        let Some(python) = python_or_skip() else { return };
        let jobs = Arc::new(JobManager::new());
        let sink = CollectingSink::new();
        let mk = |id: &'static str, secs: f64| {
            let mut cmd = Command::new(&python);
            cmd.args(["-c", &format!("import time; print('start {id}', flush=True); time.sleep({secs})")]);
            let (s1, s2, s3) = (sink.clone(), sink.clone(), sink.clone());
            let j = jobs.clone();
            jobs.spawn(
                id,
                cmd,
                JobIo {
                    on_stdout: Box::new(move |l| s1.emit("out", json!(l))),
                    on_stderr: Box::new(|_| {}),
                    on_exit: Box::new(move |_, c| {
                        s2.emit("exit", json!({ "id": id, "cancelled": c }));
                        j.finish(id);
                    }),
                },
                SpawnOpts { python: true, gpu_wait: Some(Box::new(move || s3.emit("wait", json!(id)))) },
            )
            .unwrap();
        };
        mk("g1", 1.5);
        mk("g2", 0.1);
        mk("g3", 30.0);
        // the queued jobs report that they wait (after spawn returned their ids)
        let t0 = Instant::now();
        let waited = || {
            let mut w: Vec<String> = sink.named("wait").iter().map(|v| v.as_str().unwrap().to_string()).collect();
            w.dedup();
            w
        };
        while waited().len() < 2 && t0.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(20));
        }
        let mut w = waited();
        w.sort();
        assert_eq!(w, vec!["g2", "g3"]);
        assert!(jobs.is_running("g3"), "a queued job counts as running");
        jobs.cancel("g3").unwrap(); // queued -> removed, never starts
        assert!(jobs.wait("g1", Duration::from_secs(30)) && jobs.wait("g2", Duration::from_secs(30)) && jobs.wait("g3", Duration::from_secs(30)));
        let outs: Vec<String> = sink.named("out").iter().map(|v| v.as_str().unwrap().to_string()).collect();
        assert_eq!(outs, vec!["start g1", "start g2"], "g2 starts only after g1, g3 never");
        let exits = sink.named("exit");
        assert!(exits.iter().any(|e| e["id"] == "g3" && e["cancelled"] == true));
        assert_eq!(jobs.gpu.snapshot(), (None, vec![]));
    }
}
