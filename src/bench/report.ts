import type { GpuCapabilities, PassTimings } from "../sim/pipeline"

export interface RunSummary {
  gridSize: number
  timings: PassTimings
  totalFluence: number
}

export interface BenchmarkInputs {
  caps: GpuCapabilities
  full: RunSummary | null
  coarse: RunSummary | null
  expectedFluenceFull: number
  expectedFluenceCoarse: number
  glErrors: string[]
}

// Coarse platform label for the report filename/header. No OS plugin is wired
// up yet (see COD-252 §1), so this is a best-effort UA sniff, not a real
// capability check.
export function detectPlatform(): string {
  const ua = navigator.userAgent
  if (/Windows/i.test(ua)) return "windows"
  if (/Mac OS X|Macintosh/i.test(ua)) return "macos"
  if (/Linux/i.test(ua)) return "linux"
  return "unknown"
}

function fmtMs(ms: number) {
  return `${ms.toFixed(2)} ms`
}

function fluenceRow(label: string, expected: number, measured: number | null): string {
  if (measured === null) return `  ${label}: not instrumented yet (pending COD-252 §3 CPU reference implementation)`
  const ratio = expected !== 0 ? measured / expected : NaN
  const withinOnePercent = Number.isFinite(ratio) && Math.abs(ratio - 1) <= 0.01
  return `  ${label}: expected=${expected.toExponential(4)} measured=${measured.toExponential(4)} ratio=${ratio.toFixed(4)} ${
    withinOnePercent ? "PASS" : "FAIL"
  }`
}

function runSection(label: string, run: RunSummary | null): string {
  if (!run) return `${label}: not run`
  return [
    `${label} (grid ${run.gridSize}x${run.gridSize}):`,
    `  deposit: ${fmtMs(run.timings.depositMs)}`,
    `  blur:    ${fmtMs(run.timings.blurMs)}`,
    `  ramp:    ${fmtMs(run.timings.rampMs)}`,
    `  total:   ${fmtMs(run.timings.totalMs)}`,
  ].join("\n")
}

// Builds the plain-text benchmark report described in COD-252 §1. Sections
// for stages not yet implemented (timed warm/cold series, memory/tiling
// sweep, CPU fallback timing, IPC batching) are left as explicit TODOs rather
// than fabricated numbers, so partial reports are honest about what they
// cover until COD-252 §2/§4/§5/§6 land.
export function buildReportText(inputs: BenchmarkInputs): string {
  const { caps, full, coarse, expectedFluenceFull, expectedFluenceCoarse, glErrors } = inputs
  const now = new Date()

  const lines: string[] = []
  lines.push("LaserSim benchmark report")
  lines.push(`generated: ${now.toISOString()}`)
  lines.push("")

  lines.push("== Environment ==")
  lines.push(`platform (UA sniff): ${detectPlatform()}`)
  lines.push(`user agent: ${navigator.userAgent}`)
  lines.push(`gl renderer: ${caps.renderer ?? "unavailable (WEBGL_debug_renderer_info not exposed)"}`)
  lines.push(`gl vendor: ${caps.vendor ?? "unavailable (WEBGL_debug_renderer_info not exposed)"}`)
  lines.push("app commit hash: TODO — not wired up yet (needs build-time env var)")
  lines.push("")

  lines.push("== Capabilities ==")
  lines.push(`WebGL2: ${caps.isWebGL2}`)
  lines.push(`MAX_TEXTURE_SIZE: ${caps.maxTextureSize}`)
  lines.push(`MAX_RENDERBUFFER_SIZE: ${caps.maxRenderbufferSize}`)
  lines.push(`EXT_color_buffer_float: ${caps.colorBufferFloat}`)
  lines.push(`EXT_float_blend: ${caps.floatBlend}`)
  lines.push(`OES_texture_float_linear: ${caps.textureFloatLinear}`)
  lines.push(`EXT_disjoint_timer_query_webgl2: ${caps.disjointTimerQuery}`)
  lines.push("")

  lines.push("== Timings ==")
  lines.push("method: performance.now() wall-clock around renderer.render() calls, single run (no warm-up series,")
  lines.push("no GPU sync via gl.finish()/readPixels yet — see COD-252 §2, numbers below measure submission only).")
  lines.push("")
  lines.push(runSection("Full-res", full))
  lines.push("")
  lines.push(runSection("Coarse scrub", coarse))
  lines.push("")

  lines.push("== Fluence conservation ==")
  lines.push("expected = sum(power*dt*absorption) over each job's own samples (full-res and coarse jobs sample the")
  lines.push("same path at different pitches, so each has its own discretized expectation)")
  lines.push(`  expected (full-res job): ${expectedFluenceFull.toExponential(4)}`)
  lines.push(`  expected (coarse job):   ${expectedFluenceCoarse.toExponential(4)}`)
  lines.push(fluenceRow("full-res, post-blur", expectedFluenceFull, full?.totalFluence ?? null))
  lines.push(fluenceRow("coarse, post-blur", expectedFluenceCoarse, coarse?.totalFluence ?? null))
  lines.push("  note: 'measured' is a 20mm^2 corner-window sample scaled to a grid-total estimate (see sumFluence()")
  lines.push("  in pipeline.ts), not an exact full-grid readback — treat ratios as approximate until §3 lands.")
  lines.push("  (per-stage deposit/blurH/blurV breakdown and CPU-path comparison pending COD-252 §3)")
  lines.push("")

  lines.push("== GPU memory and tiling ==")
  lines.push("  not yet implemented — pending COD-252 §4 (job-size sweep + resident-texture accounting)")
  lines.push("")

  lines.push("== CPU fallback ==")
  lines.push("  not yet implemented — pending COD-252 §5")
  lines.push("")

  lines.push("== IPC ==")
  lines.push("  not yet implemented — pending COD-252 §6 (batched streaming + status-query latency under load)")
  lines.push("")

  lines.push("== GL errors / warnings ==")
  lines.push(glErrors.length > 0 ? glErrors.join("\n") : "  none observed")

  return lines.join("\n")
}
