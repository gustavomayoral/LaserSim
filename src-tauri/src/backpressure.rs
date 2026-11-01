// Back-pressure for background threads that push many messages to the
// webview (app.emit, Channel::send).
//
// On Windows every such message is posted to the UI thread's message queue
// (tao's EventLoopProxy -> PostMessageW), and that queue holds at most
// 10,000 posted messages by default (USERPostMessageLimit). A 50k-line G-code
// burst sent one message per line outran the UI thread, filled the queue,
// and from there:
//   - PostMessageW started failing, so tauri returned "failed to send
//     message to the webview" for ~40% of the emits (dropped silently here
//     before, via `let _ =`), and
//   - the WebView2 host channel, which posts its own messages to the same
//     queue, wedged permanently: every later invoke() — including its CORS
//     preflight — went unanswered while the process stayed idle and
//     "Responding". That's the "Save report hangs" / "listener never binds"
//     symptom from COD-252's Windows testing; the 8090 per-line baseline run
//     was the trigger.
//
// Pacer bounds how far a sender can get ahead of the UI thread: every
// `every` messages it queues a marker task on the main thread and blocks
// until it runs. The proxy queue is FIFO, so by then everything sent before
// the marker has been handled.
use std::sync::mpsc::sync_channel;
use std::time::Duration;
use tauri::{AppHandle, Runtime};

/// Well under the 10,000-message limit, leaving room for WebView2's own
/// messages and input; one main-thread round trip per 500 messages is
/// negligible next to the messages themselves.
pub const DEFAULT_EVERY: u64 = 500;

// A wedged or exiting UI thread must not hang the sender thread forever.
const MAIN_THREAD_WAIT: Duration = Duration::from_secs(5);

pub struct Pacer<R: Runtime> {
    app: AppHandle<R>,
    every: u64,
    sent: u64,
    pub send_errors: u64,
}

impl<R: Runtime> Pacer<R> {
    pub fn new(app: AppHandle<R>) -> Self {
        Self { app, every: DEFAULT_EVERY, sent: 0, send_errors: 0 }
    }

    /// Call once per message sent, with that send's result.
    pub fn sent<T, E>(&mut self, result: Result<T, E>) {
        if result.is_err() {
            self.send_errors += 1;
        }
        self.sent += 1;
        if self.sent % self.every == 0 {
            self.wait_for_main_thread();
        }
    }

    fn wait_for_main_thread(&self) {
        let (tx, rx) = sync_channel(1);
        if self.app.run_on_main_thread(move || { let _ = tx.send(()); }).is_ok() {
            let _ = rx.recv_timeout(MAIN_THREAD_WAIT);
        }
    }

    /// Logs dropped sends, if any — they mean back-pressure wasn't enough.
    pub fn report(&self, label: &str) {
        if self.send_errors > 0 {
            eprintln!("{label}: {} of {} webview sends failed (UI thread queue full?)", self.send_errors, self.sent);
        }
    }
}
