//! Child-process hygiene shared by every spawner (ffmpeg, ffprobe, python).
//!
//! * Every child is started without a console window and assigned to one
//!   process-wide Windows **Job Object** created with
//!   `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. The app never closes that handle,
//!   so it is closed by the OS when Cappycat exits (normally, by a crash or
//!   by being killed) and every child still running is terminated with it —
//!   including processes the children started themselves (python's own
//!   ffmpeg children inherit the job).
//! * Python children additionally get their own nested job ([`TreeJob`]) so
//!   one pipeline run can be killed together with its descendants, and a
//!   piped stdin they watch (`CAPPYCAT_WATCH_STDIN=1`): dropping the handle
//!   asks the Python side to stop cleanly (see [`stop_python`]).
//! * [`output_timeout`] replaces `Command::output()` with a bounded wait, and
//!   [`wait_child`] waits for a shared child without holding its lock (so a
//!   concurrent cancel can always get at it to kill it).

use std::io::{self, Read};
use std::process::{Child, ChildStdin, Command, ExitStatus, Output, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// `CREATE_NO_WINDOW`: no console window flashes up for console children of a GUI app.
pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// How long a python child gets to exit on its own after its stdin was closed.
pub const PYTHON_GRACE: Duration = Duration::from_secs(2);

/// Suppress the console window of a child (no-op outside Windows).
pub fn hide_window(cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    let _ = cmd;
}

#[cfg(windows)]
mod win {
    use std::ffi::c_void;
    use std::sync::OnceLock;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, IsProcessInJob, JobObjectExtendedLimitInformation, SetInformationJobObject,
        TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    /// A job handle (plain pointer value; the handle itself is thread-safe).
    pub struct Job(pub HANDLE);
    unsafe impl Send for Job {}
    unsafe impl Sync for Job {}

    pub fn create_kill_on_close() -> Option<Job> {
        unsafe {
            let h = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if h.is_null() {
                return None;
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let ok = SetInformationJobObject(
                h,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            if ok == 0 {
                CloseHandle(h);
                return None;
            }
            Some(Job(h))
        }
    }

    static GLOBAL: OnceLock<Option<Job>> = OnceLock::new();

    pub fn global() -> Option<HANDLE> {
        GLOBAL
            .get_or_init(|| {
                let j = create_kill_on_close();
                if j.is_none() {
                    tracing::warn!("could not create the child-process job object; children may outlive the app");
                }
                j
            })
            .as_ref()
            .map(|j| j.0)
    }

    pub fn assign(job: HANDLE, process: HANDLE) -> bool {
        unsafe { AssignProcessToJobObject(job, process) != 0 }
    }

    pub fn terminate(job: HANDLE) {
        unsafe {
            TerminateJobObject(job, 1);
        }
    }

    pub fn close(job: HANDLE) {
        unsafe {
            CloseHandle(job);
        }
    }

    pub fn in_job(process: HANDLE, job: HANDLE) -> bool {
        let mut result = 0;
        unsafe { IsProcessInJob(process, job, &mut result) != 0 && result != 0 }
    }
}

/// Put a freshly spawned process into the process-wide kill-on-close job.
/// Returns false when that is not possible (non-Windows, or the OS refused).
pub fn adopt(child: &Child) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        adopt_raw(child.as_raw_handle())
    }
    #[cfg(not(windows))]
    {
        let _ = child;
        false
    }
}

/// [`adopt`] for a raw process handle (e.g. a `tokio::process::Child`).
#[cfg(windows)]
pub fn adopt_raw(handle: std::os::windows::io::RawHandle) -> bool {
    match win::global() {
        Some(job) => {
            let ok = win::assign(job, handle as _);
            if !ok {
                tracing::debug!("AssignProcessToJobObject failed: {}", io::Error::last_os_error());
            }
            ok
        }
        None => false,
    }
}

/// Is the child a member of the process-wide job? (tests / diagnostics)
pub fn is_adopted(child: &Child) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        match win::global() {
            Some(job) => win::in_job(child.as_raw_handle() as _, job),
            None => false,
        }
    }
    #[cfg(not(windows))]
    {
        let _ = child;
        false
    }
}

/// Spawn a child with the hygiene rules above (hidden window + global job).
pub fn spawn(cmd: &mut Command) -> io::Result<Child> {
    hide_window(cmd);
    let child = cmd.spawn()?;
    adopt(&child);
    Ok(child)
}

/// A nested job holding one process tree (a python child and whatever it starts).
/// [`TreeJob::kill`] terminates the whole tree; dropping it does too (kill-on-close),
/// which only matters for descendants still running after the child itself ended.
pub struct TreeJob {
    #[cfg(windows)]
    job: Option<win::Job>,
}

impl TreeJob {
    /// Create the tree job and put `child` in it (after the global job, so the
    /// tree job nests inside it).
    pub fn for_child(child: &Child) -> Self {
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            let job = win::create_kill_on_close().filter(|j| win::assign(j.0, child.as_raw_handle() as _));
            Self { job }
        }
        #[cfg(not(windows))]
        {
            let _ = child;
            Self {}
        }
    }

    /// Terminate every process in the tree.
    pub fn kill(&self) {
        #[cfg(windows)]
        if let Some(j) = &self.job {
            win::terminate(j.0);
        }
    }
}

impl Drop for TreeJob {
    fn drop(&mut self) {
        #[cfg(windows)]
        if let Some(j) = self.job.take() {
            win::close(j.0);
        }
    }
}

/// Wait for a shared child without holding its lock for longer than a `try_wait`.
pub fn wait_child(child: &Mutex<Child>) -> io::Result<ExitStatus> {
    let mut sleep = Duration::from_millis(1);
    loop {
        {
            let mut c = child.lock().map_err(|_| io::Error::other("child lock poisoned"))?;
            if let Some(status) = c.try_wait()? {
                return Ok(status);
            }
        }
        std::thread::sleep(sleep);
        sleep = (sleep * 2).min(Duration::from_millis(25));
    }
}

/// Like [`wait_child`] but gives up after `timeout` (returns `Ok(None)`).
pub fn wait_child_timeout(child: &Mutex<Child>, timeout: Duration) -> io::Result<Option<ExitStatus>> {
    let start = Instant::now();
    loop {
        {
            let mut c = child.lock().map_err(|_| io::Error::other("child lock poisoned"))?;
            if let Some(status) = c.try_wait()? {
                return Ok(Some(status));
            }
        }
        if start.elapsed() >= timeout {
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Stop a python child the polite way: close its stdin (the pipeline's watchdog
/// then exits and stops its own ffmpeg children), give it [`PYTHON_GRACE`],
/// then kill it (and its tree). Blocks for at most the grace period.
pub fn stop_python(child: &Mutex<Child>, stdin: Option<ChildStdin>, tree: Option<&TreeJob>) {
    drop(stdin);
    if matches!(wait_child_timeout(child, PYTHON_GRACE), Ok(Some(_))) {
        // exited on its own; make sure no grandchild lingers either
        if let Some(t) = tree {
            t.kill();
        }
        return;
    }
    if let Some(t) = tree {
        t.kill();
    }
    if let Ok(mut c) = child.lock() {
        let _ = c.kill();
    }
}

/// `Command::output()` with the hygiene rules and a timeout: stdin is null,
/// stdout / stderr are drained on threads (so neither pipe can fill up and
/// deadlock), and the child is killed when `timeout` passes
/// (`ErrorKind::TimedOut`).
pub fn output_timeout(cmd: &mut Command, timeout: Duration) -> io::Result<Output> {
    output_impl(cmd, timeout, false)
}

/// [`output_timeout`] for a `python -m cappycat_pipeline …` child: its stdin is piped
/// and held open until it exits (`CAPPYCAT_WATCH_STDIN=1` must be set by the caller),
/// and on timeout it is stopped with [`stop_python`] (process tree included).
pub fn output_python(cmd: &mut Command, timeout: Duration) -> io::Result<Output> {
    output_impl(cmd, timeout, true)
}

fn output_impl(cmd: &mut Command, timeout: Duration, python: bool) -> io::Result<Output> {
    cmd.stdin(if python { Stdio::piped() } else { Stdio::null() }).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = spawn(cmd)?;
    let stdin = child.stdin.take();
    let tree = python.then(|| TreeJob::for_child(&child));
    let out = child.stdout.take();
    let err = child.stderr.take();
    let read = |p: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(mut p) = p {
                let _ = p.read_to_end(&mut buf);
            }
            buf
        })
    };
    let t_out = read(out.map(|p| Box::new(p) as Box<dyn Read + Send>));
    let t_err = read(err.map(|p| Box::new(p) as Box<dyn Read + Send>));
    let child = Mutex::new(child);
    let status = match wait_child_timeout(&child, timeout)? {
        Some(s) => s,
        None => {
            if python {
                stop_python(&child, stdin, tree.as_ref());
            }
            let mut c = child.lock().unwrap();
            let _ = c.kill();
            let _ = c.wait();
            drop(c);
            let _ = t_out.join();
            let _ = t_err.join();
            return Err(io::Error::new(io::ErrorKind::TimedOut, format!("timed out after {:.0} s", timeout.as_secs_f64())));
        }
    };
    drop(stdin);
    if let Some(t) = &tree {
        t.kill(); // leftover grandchildren would keep the pipes open
    }
    let stdout = t_out.join().unwrap_or_default();
    let stderr = t_err.join().unwrap_or_default();
    Ok(Output { status, stdout, stderr })
}

/// A shared child plus (for python) its stdin and process-tree job.
pub struct ManagedChild {
    pub child: Arc<Mutex<Child>>,
    pub stdin: Mutex<Option<ChildStdin>>,
    pub tree: Option<TreeJob>,
}

impl ManagedChild {
    /// Stop the child: python children get the stdin-EOF grace period (run this on a
    /// background thread), everything else is killed immediately.
    pub fn stop(&self) {
        let stdin = self.stdin.lock().ok().and_then(|mut s| s.take());
        if stdin.is_some() || self.tree.is_some() {
            stop_python(&self.child, stdin, self.tree.as_ref());
        } else if let Ok(mut c) = self.child.lock() {
            let _ = c.kill();
        }
    }

    pub fn is_python(&self) -> bool {
        self.tree.is_some() || self.stdin.lock().map(|s| s.is_some()).unwrap_or(false)
    }

    pub fn has_exited(&self) -> bool {
        self.child.try_lock().map(|mut c| !matches!(c.try_wait(), Ok(None))).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn children_join_the_kill_on_close_job() {
        let Some(bins) = crate::ffmpeg::find_binaries().ok() else {
            eprintln!("SKIP: ffmpeg not available");
            return;
        };
        let mut cmd = Command::new(&bins.ffmpeg);
        cmd.args(["-hide_banner", "-f", "lavfi", "-i", "anullsrc", "-t", "30", "-f", "null", "-"]);
        cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
        let mut child = spawn(&mut cmd).unwrap();
        if cfg!(windows) {
            assert!(is_adopted(&child), "spawned children must be in the process-wide job object");
        }
        child.kill().unwrap();
        child.wait().unwrap();
    }

    #[test]
    fn output_timeout_kills_slow_children_and_drains_pipes() {
        let Some(bins) = crate::ffmpeg::find_binaries().ok() else {
            eprintln!("SKIP: ffmpeg not available");
            return;
        };
        // a lot of stderr (would deadlock a naive reader) + a normal exit
        let mut cmd = Command::new(&bins.ffmpeg);
        cmd.args(["-v", "debug", "-f", "lavfi", "-i", "testsrc=duration=1:size=64x48:rate=24", "-f", "null", "-"]);
        let out = output_timeout(&mut cmd, Duration::from_secs(60)).unwrap();
        assert!(out.status.success());
        assert!(out.stderr.len() > 10_000, "stderr drained: {} bytes", out.stderr.len());
        // endless input → killed by the timeout
        let mut cmd = Command::new(&bins.ffmpeg);
        cmd.args(["-v", "error", "-re", "-f", "lavfi", "-i", "anullsrc", "-f", "null", "-"]);
        let t = Instant::now();
        let err = output_timeout(&mut cmd, Duration::from_millis(500)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
        assert!(t.elapsed() < Duration::from_secs(10));
    }
}
