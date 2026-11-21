//! Keeps runtime threads out of Python once the interpreter is going away.
//!
//! The tokio runtime outlives the interpreter: its threads are still running
//! while Python finalizes, and attaching to a finalized interpreter panics in
//! `pyo3`. Two things on those threads attach: the `log` bridge, for every Rust
//! log record, and the step that hands a finished future's result back to
//! asyncio. Both go through [`attach_if_alive`], which does nothing once the
//! interpreter has started to exit.

use pyo3::prelude::*;
use pyo3_async_runtimes::TaskLocals;
use pyo3_async_runtimes::generic::{self, ContextExt, Runtime};
use std::cell::OnceCell;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::task;

/// Set by an `atexit` hook, which Python runs before it starts to finalize.
static EXITING: AtomicBool = AtomicBool::new(false);

/// Runs `f` attached to the interpreter, or not at all if Python is exiting.
///
/// The flag covers an orderly exit completely: it is set while the interpreter
/// is still whole. [`Python::try_attach`] covers the rest, such as an exit
/// that skips `atexit`.
pub fn attach_if_alive<F, R>(f: F) -> Option<R>
where
    F: for<'py> FnOnce(Python<'py>) -> R,
{
    if EXITING.load(Ordering::Acquire) {
        return None;
    }
    Python::try_attach(f)
}

#[pyfunction]
fn _mark_interpreter_exiting() {
    EXITING.store(true, Ordering::Release);
}

/// Registers the `atexit` hook that turns [`attach_if_alive`] off.
pub fn register_exit_hook(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let hook = wrap_pyfunction!(_mark_interpreter_exiting, m)?;
    m.py().import("atexit")?.call_method1("register", (hook,))?;
    Ok(())
}

/// The Python logging bridge, silent once the interpreter is exiting.
pub struct ExitSafeLogger(pub pyo3_log::Logger);

impl log::Log for ExitSafeLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        self.0.enabled(metadata)
    }

    fn log(&self, record: &log::Record<'_>) {
        // Checked here so a record Python would drop never takes the
        // interpreter lock; the bridge checks again on its own.
        if self.0.enabled(record.metadata()) {
            // Already attached inside the closure, so the bridge's own attach
            // cannot reach the interpreter check that panics.
            attach_if_alive(|_| self.0.log(record));
        }
    }

    fn flush(&self) {
        self.0.flush();
    }
}

tokio::task_local! {
    static TASK_LOCALS: OnceCell<TaskLocals>;
}

/// The shared tokio runtime, with completions that tolerate interpreter exit.
///
/// Identical to `pyo3_async_runtimes::tokio` except for `spawn_blocking`,
/// which that crate uses to deliver a future's result to asyncio.
struct ExitSafeTokio;

impl Runtime for ExitSafeTokio {
    type JoinError = task::JoinError;
    type JoinHandle = task::JoinHandle<()>;

    fn spawn<F>(fut: F) -> Self::JoinHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        pyo3_async_runtimes::tokio::get_runtime().spawn(fut)
    }

    fn spawn_blocking<F>(f: F) -> Self::JoinHandle
    where
        F: FnOnce() + Send + 'static,
    {
        pyo3_async_runtimes::tokio::get_runtime().spawn_blocking(move || {
            attach_if_alive(move |_| f());
        })
    }
}

impl ContextExt for ExitSafeTokio {
    fn scope<F, R>(locals: TaskLocals, fut: F) -> Pin<Box<dyn Future<Output = R> + Send>>
    where
        F: Future<Output = R> + Send + 'static,
    {
        Box::pin(TASK_LOCALS.scope(OnceCell::from(locals), fut))
    }

    fn get_task_locals() -> Option<TaskLocals> {
        TASK_LOCALS
            .try_with(|cell| cell.get().cloned())
            .unwrap_or_default()
    }
}

/// Runs `fut` on the shared runtime and returns the asyncio future for it.
///
/// `T` is converted to a Python object only when the result is delivered, on a
/// thread that is attached, so `fut` itself must not attach.
pub fn future_into_py<F, T>(py: Python<'_>, fut: F) -> PyResult<Bound<'_, PyAny>>
where
    F: Future<Output = PyResult<T>> + Send + 'static,
    T: for<'py> IntoPyObject<'py> + Send + 'static,
{
    generic::future_into_py::<ExitSafeTokio, F, T>(py, fut)
}
