//! Logging: a daily-rolling log file in `%LOCALAPPDATA%\cappycat\logs\`
//! (`cappycat.YYYY-MM-DD.log` for the app, `cappycat-cli.YYYY-MM-DD.log` for
//! the CLI; the last 14 files are kept) plus the console.
//!
//! Levels: the file gets `info` and up (`CAPPYCAT_LOG` overrides, e.g.
//! `CAPPYCAT_LOG=debug`); the console level is chosen by the caller (the app
//! uses `info`, which only shows in debug builds that have a console; the CLI
//! uses `warn`, or `info` with `-v`). Panics are logged too.

use std::path::PathBuf;
use std::sync::OnceLock;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer, Registry};

static GUARD: OnceLock<tracing_appender::non_blocking::WorkerGuard> = OnceLock::new();

/// `%LOCALAPPDATA%\Cappycat\logs` (see [`crate::paths`]).
pub fn logs_dir() -> PathBuf {
    crate::paths::logs_dir()
}

/// Install the global subscriber (file + console). Safe to call more than once
/// (later calls do nothing). Returns the log folder when the file log is active.
pub fn init(file_prefix: &str, console_level: &str) -> Option<PathBuf> {
    type Boxed = Box<dyn Layer<Registry> + Send + Sync>;
    let dir = logs_dir();
    let mut layers: Vec<Boxed> = vec![tracing_subscriber::fmt::layer()
        .with_target(false)
        .with_writer(std::io::stderr)
        .with_filter(EnvFilter::new(console_level))
        .boxed()];
    let appender = std::fs::create_dir_all(&dir).ok().and_then(|_| {
        tracing_appender::rolling::Builder::new()
            .rotation(tracing_appender::rolling::Rotation::DAILY)
            .filename_prefix(file_prefix)
            .filename_suffix("log")
            .max_log_files(14)
            .build(&dir)
            .ok()
    });
    let mut guard = None;
    if let Some(appender) = appender {
        let (writer, g) = tracing_appender::non_blocking(appender);
        let filter = EnvFilter::try_from_env("CAPPYCAT_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
        layers.push(tracing_subscriber::fmt::layer().with_ansi(false).with_writer(writer).with_filter(filter).boxed());
        guard = Some(g);
    }
    let installed = tracing_subscriber::registry().with(layers).try_init().is_ok();
    install_panic_hook();
    match guard {
        Some(g) if installed => {
            let _ = GUARD.set(g);
            Some(dir)
        }
        _ => None,
    }
}

/// Log panics (release builds have no console) before the default hook runs.
fn install_panic_hook() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let thread = std::thread::current().name().unwrap_or("?").to_string();
            tracing::error!("panic in thread '{thread}': {info}");
            default(info);
        }));
    });
}
