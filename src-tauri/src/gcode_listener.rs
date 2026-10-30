// COD-251 "also check": a minimal TCP listener that accepts a line-oriented
// G-code stream and forwards each line to the webview as a Tauri event, so
// we can measure whether IPC is a bottleneck when only G-code (not simulation
// data) crosses the native/webview boundary. No GRBL protocol, no motion
// planning — that's explicitly out of scope for this spike.
use serde::Serialize;
use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter};

pub const DEFAULT_PORT: u16 = 8090;

#[derive(Clone, Serialize)]
struct GcodeLine {
    seq: u64,
    line: String,
    rust_ms: f64,
}

#[derive(Clone, Serialize)]
struct GcodeDone {
    seq: u64,
}

fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
        * 1000.0
}

/// Spawns a background thread listening on 127.0.0.1:<port>. Each accepted
/// connection is read line by line; every line becomes a `gcode-line` event,
/// and the connection closing emits `gcode-done` with the total count so the
/// frontend can compute received-lines-per-second.
pub fn spawn(app: AppHandle, port: u16) {
    thread::spawn(move || {
        let listener = match TcpListener::bind(("127.0.0.1", port)) {
            Ok(l) => l,
            Err(err) => {
                eprintln!("gcode_listener: failed to bind 127.0.0.1:{port}: {err}");
                return;
            }
        };
        println!("gcode_listener: listening on 127.0.0.1:{port}");

        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let app = app.clone();
            thread::spawn(move || {
                let reader = BufReader::new(stream);
                let seq = AtomicU64::new(0);
                for line in reader.lines() {
                    let Ok(line) = line else { break };
                    let n = seq.fetch_add(1, Ordering::Relaxed) + 1;
                    let _ = app.emit(
                        "gcode-line",
                        GcodeLine {
                            seq: n,
                            line,
                            rust_ms: now_ms(),
                        },
                    );
                }
                let total = seq.load(Ordering::Relaxed);
                let _ = app.emit("gcode-done", GcodeDone { seq: total });
            });
        }
    });
}
