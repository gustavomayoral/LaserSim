// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
mod gcode_listener;

use tauri::Manager;

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

// Writes a benchmark report (plain text, already formatted by the JS side)
// to the Desktop so it's easy to find and attach to the Linear issue. Returns
// the full path written, for the UI to display.
#[tauri::command]
fn write_benchmark_report(app: tauri::AppHandle, platform: String, report: String) -> Result<String, String> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let desktop_dir = app.path().desktop_dir().map_err(|e| e.to_string())?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_secs();
    let filename = format!("lasersim-bench-{platform}-{timestamp}.txt");
    let path = desktop_dir.join(&filename);
    std::fs::write(&path, report).map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().into_owned())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet, write_benchmark_report])
        .setup(|app| {
            gcode_listener::spawn(app.handle().clone(), gcode_listener::DEFAULT_PORT);
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
