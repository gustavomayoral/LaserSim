// IPC benchmark listeners (COD-252 §6): re-runs the COD-251 line-stream
// check with lines batched per message, and again over a Tauri v2 Channel
// instead of a global event, so the two alternatives to "one global event
// per line" (gcode_listener.rs's baseline) can be compared on the same
// machine. Also exposes a trivial `ping` command so the frontend can
// measure invoke() round-trip latency while a stream is in flight.
use serde::Serialize;
use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::net::{TcpListener, TcpStream};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::ipc::Channel;
use tauri::{AppHandle, Emitter};

#[derive(Clone, Serialize)]
pub struct GcodeBatch {
    pub seq_start: u64,
    pub count: u32,
    pub lines: Vec<String>,
    pub rust_ms: f64,
}

#[derive(Clone, Serialize)]
pub struct GcodeBatchDone {
    pub total: u64,
}

fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
        * 1000.0
}

fn bound_ports() -> &'static Mutex<HashSet<u16>> {
    static PORTS: OnceLock<Mutex<HashSet<u16>>> = OnceLock::new();
    PORTS.get_or_init(|| Mutex::new(HashSet::new()))
}

// Binds synchronously so a failure (already bound, port in use) reaches the
// frontend as a rejected invoke() instead of a silently-dead listener.
fn bind_once(port: u16) -> Result<TcpListener, String> {
    let mut bound = bound_ports().lock().map_err(|e| e.to_string())?;
    if bound.contains(&port) {
        return Err(format!("a listener is already bound to 127.0.0.1:{port}"));
    }
    let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| e.to_string())?;
    bound.insert(port);
    Ok(listener)
}

// Each "Listen for ... run" click is good for exactly one sender connection:
// accept it, close the listener and free the port, then stream that
// connection. Keeping the listener alive across runs made every button
// single-use per app launch — the second click hit bind_once's "already
// bound" error, while the first listener kept accepting connections whose
// events no frontend session was subscribed to anymore (and, for the
// channel variant, kept sending into a stale Channel).
fn serve_one_run(listener: TcpListener, port: u16, handle: impl FnOnce(TcpStream) + Send + 'static) {
    thread::spawn(move || {
        let accepted = listener.accept();
        drop(listener);
        if let Ok(mut bound) = bound_ports().lock() {
            bound.remove(&port);
        }
        match accepted {
            Ok((stream, _)) => handle(stream),
            Err(err) => eprintln!("ipc_bench: accept on 127.0.0.1:{port} failed: {err}"),
        }
    });
}

/// Batched-global-event variant of gcode_listener::spawn: lines are
/// accumulated into a Vec and emitted as one "gcode-batch" event per
/// `batch_size` lines (plus a final partial batch), instead of one event
/// per line.
#[tauri::command]
pub fn start_gcode_batched(app: AppHandle, port: u16, batch_size: usize) -> Result<(), String> {
    let listener = bind_once(port)?;
    let batch_size = batch_size.max(1);
    serve_one_run(listener, port, move |stream| {
        let reader = BufReader::new(stream);
        let mut seq: u64 = 0;
        let mut batch: Vec<String> = Vec::with_capacity(batch_size);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            seq += 1;
            batch.push(line);
            if batch.len() >= batch_size {
                let seq_start = seq - batch.len() as u64 + 1;
                let _ = app.emit(
                    "gcode-batch",
                    GcodeBatch { seq_start, count: batch.len() as u32, lines: std::mem::take(&mut batch), rust_ms: now_ms() },
                );
            }
        }
        if !batch.is_empty() {
            let seq_start = seq - batch.len() as u64 + 1;
            let _ = app.emit(
                "gcode-batch",
                GcodeBatch { seq_start, count: batch.len() as u32, lines: batch, rust_ms: now_ms() },
            );
        }
        let _ = app.emit("gcode-batch-done", GcodeBatchDone { total: seq });
    });
    Ok(())
}

/// Same batching as start_gcode_batched, but delivered through a Tauri v2
/// Channel (passed in from the frontend's invoke call) instead of a global
/// `app.emit`, so the two IPC transports can be compared directly.
#[tauri::command]
pub fn start_gcode_channel(port: u16, batch_size: usize, channel: Channel<GcodeBatch>) -> Result<(), String> {
    let listener = bind_once(port)?;
    let batch_size = batch_size.max(1);
    serve_one_run(listener, port, move |stream| {
        let reader = BufReader::new(stream);
        let mut seq: u64 = 0;
        let mut batch: Vec<String> = Vec::with_capacity(batch_size);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            seq += 1;
            batch.push(line);
            if batch.len() >= batch_size {
                let seq_start = seq - batch.len() as u64 + 1;
                let _ = channel.send(GcodeBatch {
                    seq_start,
                    count: batch.len() as u32,
                    lines: std::mem::take(&mut batch),
                    rust_ms: now_ms(),
                });
            }
        }
        if !batch.is_empty() {
            let seq_start = seq - batch.len() as u64 + 1;
            let _ = channel.send(GcodeBatch { seq_start, count: batch.len() as u32, lines: batch, rust_ms: now_ms() });
        }
        // Channel has no separate "done" event type, so send a
        // zero-count sentinel the frontend can use to know the
        // stream ended (mirrors gcode-batch-done for the event path).
        let _ = channel.send(GcodeBatch { seq_start: seq + 1, count: 0, lines: Vec::new(), rust_ms: now_ms() });
    });
    Ok(())
}

/// Trivial round-trip-latency probe for the COD-252 §6 "`?` status query
/// while a stream is in flight" check: the frontend times invoke("ping")
/// itself, so all this needs to do is return immediately.
#[tauri::command]
pub fn ping() -> f64 {
    now_ms()
}
