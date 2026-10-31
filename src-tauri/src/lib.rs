// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
mod gcode_listener;
mod ipc_bench;

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

// Default filename offered by the frontend's save dialog (App.tsx), kept
// here too so both sides agree on the naming convention without the
// frontend needing to duplicate the platform-sniffing/timestamp logic.
#[tauri::command]
fn default_benchmark_report_name(platform: String) -> Result<String, String> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_secs();
    Ok(format!("lasersim-bench-{platform}-{timestamp}.txt"))
}

// Writes a benchmark report (plain text, already formatted by the JS side)
// to a path the user picked via the frontend's save dialog. Returns the
// path written, for the UI to display.
#[tauri::command]
fn write_benchmark_report(path: String, report: String) -> Result<String, String> {
    std::fs::write(&path, report).map_err(|e| e.to_string())?;
    Ok(path)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            greet,
            default_benchmark_report_name,
            write_benchmark_report,
            ipc_bench::start_gcode_batched,
            ipc_bench::start_gcode_channel,
            ipc_bench::ping
        ])
        .setup(|app| {
            gcode_listener::spawn(app.handle().clone(), gcode_listener::DEFAULT_PORT);
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
