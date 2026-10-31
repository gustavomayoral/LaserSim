// Environment fields for the benchmark report's "Environment" section
// (COD-252 §1) that the webview can't see for itself: the real OS name and
// build (the user agent freezes Windows at "Windows NT 10.0" even on
// Windows 11), the webview runtime version (the UA's "Edg/…" is the
// browser brand, not the WebView2 runtime), and the app's commit.
use serde::Serialize;
use std::path::Path;
use std::process::Command;

#[derive(Serialize)]
pub struct EnvironmentInfo {
    /// "windows" / "macos" / "linux" — also used in the report filename.
    platform: &'static str,
    arch: &'static str,
    /// e.g. "Windows 11 24H2 (build 10.0.26300.1234)"; None if probing failed.
    os: Option<String>,
    webview_version: Option<String>,
    /// Short hash of the checkout this dev build runs from, with "-dirty" if
    /// tracked files have uncommitted changes. Read at runtime (not baked in
    /// at compile time) so it stays right across `tauri dev` reloads; None
    /// when not running from a git checkout (e.g. an installed build).
    commit: Option<String>,
}

pub fn platform() -> &'static str {
    std::env::consts::OS
}

#[tauri::command]
pub fn environment_info() -> EnvironmentInfo {
    EnvironmentInfo {
        platform: platform(),
        arch: std::env::consts::ARCH,
        os: os_description(),
        webview_version: tauri::webview_version().ok(),
        commit: git_commit(),
    }
}

// Runs a probe command without flashing a console window (a GUI-subsystem
// app spawning cmd/git on Windows otherwise opens one per call).
fn run(program: &str, args: &[&str]) -> Option<String> {
    let mut cmd = Command::new(program);
    cmd.args(args);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let output = cmd.output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[cfg(windows)]
fn os_description() -> Option<String> {
    // `ver` prints e.g. "Microsoft Windows [Version 10.0.26300.1234]".
    let ver = run("cmd", &["/c", "ver"])?;
    let version = ver.split("Version ").nth(1)?.trim_end_matches(']').trim().to_string();
    let build: u32 = version.split('.').nth(2)?.parse().ok()?;
    // Windows 11 still reports major version 10; build 22000+ is 11.
    let name = if build >= 22000 { "Windows 11" } else { "Windows 10" };
    let display_version = run(
        "reg",
        &["query", r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion", "/v", "DisplayVersion"],
    )
    .and_then(|out| out.split_whitespace().last().map(str::to_string));
    Some(match display_version {
        Some(dv) => format!("{name} {dv} (build {version})"),
        None => format!("{name} (build {version})"),
    })
}

#[cfg(target_os = "macos")]
fn os_description() -> Option<String> {
    let name = run("sw_vers", &["-productName"])?;
    let version = run("sw_vers", &["-productVersion"])?;
    let build = run("sw_vers", &["-buildVersion"])?;
    Some(format!("{name} {version} (build {build})"))
}

#[cfg(not(any(windows, target_os = "macos")))]
fn os_description() -> Option<String> {
    let release = std::fs::read_to_string("/etc/os-release").ok()?;
    let pretty = release
        .lines()
        .find_map(|l| l.strip_prefix("PRETTY_NAME="))
        .map(|v| v.trim_matches('"').to_string())?;
    let kernel = run("uname", &["-r"]).unwrap_or_default();
    Some(format!("{pretty} (kernel {kernel})"))
}

fn git_commit() -> Option<String> {
    // Same compile-time repo root as lib.rs's session log path.
    let repo = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/.."));
    if !repo.join(".git").exists() {
        return None;
    }
    let repo = repo.to_str()?;
    // safe.directory=*: the Windows test machine's checkout is owned by
    // Administrators, which makes plain git refuse to run there.
    let git = |args: &[&str]| {
        let mut full = vec!["-c", "safe.directory=*", "-C", repo];
        full.extend_from_slice(args);
        run("git", &full)
    };
    let hash = git(&["rev-parse", "--short", "HEAD"])?;
    let dirty = git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
    Some(if dirty { format!("{hash}-dirty") } else { hash })
}
