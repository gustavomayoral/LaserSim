// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
mod backpressure;
mod env_info;
mod gcode_listener;
mod ipc_bench;

use std::io::Write;
use tauri::Manager;

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

// Mirrors the frontend's in-app status log to a plain-text file at the
// project root (one level up from src-tauri, where this crate's manifest
// lives — resolved at compile time so it doesn't depend on the process's
// working directory), so it can be read without the app's own window
// (remote debugging, or just tailing it from a terminal). Truncated once
// at startup (see run()) so each launch starts a fresh file instead of an
// ever-growing one across relaunches.
fn session_log_path() -> std::path::PathBuf {
    std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/..")).join("lasersim-session.log")
}

#[tauri::command]
fn append_session_log(line: String) -> Result<(), String> {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(session_log_path())
        .map_err(|e| e.to_string())?;
    writeln!(file, "{line}").map_err(|e| e.to_string())
}

// Default full path offered to the frontend's editable save-path field
// (App.tsx) — Desktop plus a timestamped filename. Not a native save
// dialog: tauri-plugin-dialog's save() hung indefinitely on at least one
// real Windows/WebView2 machine (no dialog ever appeared, including after
// checking the taskbar/Alt+Tab — not just an unfocused-window issue), so
// the frontend uses a plain text input instead; this only computes the
// starting value for it.
#[tauri::command]
fn default_benchmark_report_path(app: tauri::AppHandle) -> Result<String, String> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let desktop_dir = app.path().desktop_dir().map_err(|e| e.to_string())?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_secs();
    // Platform in the name (COD-252 §1) so Windows and macOS reports sit
    // side by side without renaming.
    let filename = format!("lasersim-bench-{}-{timestamp}.txt", env_info::platform());
    Ok(desktop_dir.join(filename).to_string_lossy().into_owned())
}

// Writes a benchmark report (plain text, already formatted by the JS side)
// to whatever path the frontend's editable save-path field currently holds.
// Returns the path written, for the UI to display.
#[tauri::command]
fn write_benchmark_report(path: String, report: String) -> Result<String, String> {
    std::fs::write(&path, report).map_err(|e| e.to_string())?;
    Ok(path)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            greet,
            default_benchmark_report_path,
            write_benchmark_report,
            append_session_log,
            env_info::environment_info,
            ipc_bench::start_gcode_batched,
            ipc_bench::start_gcode_channel,
            ipc_bench::ping
        ])
        .setup(|app| {
            let _ = std::fs::write(session_log_path(), "");
            gcode_listener::spawn(app.handle().clone(), gcode_listener::DEFAULT_PORT);
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
