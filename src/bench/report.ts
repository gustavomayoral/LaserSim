import type { GpuCapabilities, PassTimings } from "../sim/pipeline"
import { minMaxMedian } from "../sim/gpuTimer"
import type { MemorySweepStep } from "./memorySweep"
import type { CpuFallbackResult } from "./cpuFallback"
import type { IpcModeResult } from "./ipcBench"

export interface RunSummary {
  gridSize: number
  timings: PassTimings
  totalFluence: number
}

export interface TimingSeries {
  cold: PassTimings
  warm: PassTimings[]
}

export interface BenchmarkInputs {
  caps: GpuCapabilities
  full: RunSummary | null
  coarse: RunSummary | null
  expectedFluenceFull: number
  expectedFluenceCoarse: number
  glErrors: string[]
  fullSeries: TimingSeries | null
  coarseSeries: TimingSeries | null
  memorySweep: MemorySweepStep[] | null
  cpuFallback: CpuFallbackResult | null
  ipcResults: IpcModeResult[] | null
  // Unsynced (submission-time-only) coarse-scrub total, as an estimate of
  // what real interactive use would cost — NOT comparable to the synced
  // numbers above. See pipeline.ts's run() `sync` param.
  coarseUnsyncedTotalMs: number | null
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

function fmtStage(label: string, wallMs: number, gpuMs: number | null): string {
  const gpuPart = gpuMs !== null ? `gpu=${fmtMs(gpuMs)}` : "gpu=unavailable"
  return `  ${label}: wall=${fmtMs(wallMs)} ${gpuPart}`
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
    fmtStage("deposit", run.timings.deposit.wallMs, run.timings.deposit.gpuMs),
    fmtStage("blur H ", run.timings.blurH.wallMs, run.timings.blurH.gpuMs),
    fmtStage("blur V ", run.timings.blurV.wallMs, run.timings.blurV.gpuMs),
    fmtStage("ramp   ", run.timings.ramp.wallMs, run.timings.ramp.gpuMs),
    `  total:   ${fmtMs(run.timings.totalMs)}`,
  ].join("\n")
}

const STAGES = ["deposit", "blurH", "blurV", "ramp"] as const

// Reports cold (first-run, possibly still paying one-time allocation cost
// even after pipeline warmup) and min/median/max across `warm` runs per
// stage (COD-252 §1/§2).
function seriesSection(label: string, series: TimingSeries | null): string {
  if (!series) return `${label}: not run`
  const lines = [`${label} (${series.warm.length} warm runs after 1 cold run):`]
  lines.push(`  cold total: ${fmtMs(series.cold.totalMs)}`)
  for (const stage of STAGES) {
    const coldWall = series.cold[stage].wallMs
    const warmWall = series.warm.map((t) => t[stage].wallMs)
    const { min, median, max } = minMaxMedian(warmWall)
    lines.push(
      `  ${stage.padEnd(7)}: cold=${fmtMs(coldWall)}  warm min/median/max = ${fmtMs(min)} / ${fmtMs(median)} / ${fmtMs(max)}`,
    )
  }
  const warmTotal = series.warm.map((t) => t.totalMs)
  const { min, median, max } = minMaxMedian(warmTotal)
  lines.push(`  total  : cold=${fmtMs(series.cold.totalMs)}  warm min/median/max = ${fmtMs(min)} / ${fmtMs(median)} / ${fmtMs(max)}`)
  return lines.join("\n")
}

// Builds the plain-text benchmark report described in COD-252 §1. Sections
// for stages not yet implemented (timed warm/cold series, memory/tiling
// sweep, CPU fallback timing, IPC batching) are left as explicit TODOs rather
// than fabricated numbers, so partial reports are honest about what they
// cover until COD-252 §2/§4/§5/§6 land.
export function buildReportText(inputs: BenchmarkInputs): string {
  const {
    caps,
    full,
    coarse,
    expectedFluenceFull,
    expectedFluenceCoarse,
    glErrors,
    fullSeries,
    coarseSeries,
    memorySweep,
    cpuFallback,
    ipcResults,
    coarseUnsyncedTotalMs,
  } = inputs
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
  lines.push("method: each stage is synced before its wall-clock end is taken (gl.finish() + 1px readPixels — see")
  lines.push("gpuTimer.ts), so wall times reflect completed GPU work, not just submission (COD-252 §2). gpu= times")
  lines.push("come from EXT_disjoint_timer_query_webgl2 when available (null otherwise — see Capabilities above).")
  lines.push("Shaders are warmed up once at pipeline construction, before any of the runs below.")
  lines.push("")
  lines.push(runSection("Full-res (single run, shown in UI)", full))
  lines.push("")
  lines.push(runSection("Coarse scrub (single run, shown in UI)", coarse))
  lines.push("")
  lines.push(seriesSection("Full-res series", fullSeries))
  lines.push("")
  lines.push(seriesSection("Coarse scrub series", coarseSeries))
  lines.push("")
  if (coarseUnsyncedTotalMs !== null) {
    lines.push(
      `Coarse scrub, UNSYNCED (interactive-use estimate, NOT GPU-complete — see pipeline.ts's run() sync param): ${fmtMs(
        coarseUnsyncedTotalMs,
      )}`,
    )
    lines.push(
      "  this is submission time only (COD-251's original method); real interactive use (e.g. a live scrub",
    )
    lines.push(
      "  control) wouldn't force the synchronous gl.finish()/readPixels the benchmark numbers above need for",
    )
    lines.push("  correctness, so this is the number closer to what that would actually feel like.")
  } else {
    lines.push("Coarse scrub, UNSYNCED (interactive-use estimate): not measured this run")
  }
  lines.push("")

  lines.push("== Fluence conservation ==")
  lines.push("expected = sum(power*dt*absorption) over each job's own samples (full-res and coarse jobs sample the")
  lines.push("same path at different pitches, so each has its own discretized expectation)")
  lines.push(`  expected (full-res job): ${expectedFluenceFull.toExponential(4)}`)
  lines.push(`  expected (coarse job):   ${expectedFluenceCoarse.toExponential(4)}`)
  lines.push(fluenceRow("full-res, post-blur", expectedFluenceFull, full?.totalFluence ?? null))
  lines.push(fluenceRow("coarse, post-blur", expectedFluenceCoarse, coarse?.totalFluence ?? null))
  lines.push("  note: 'measured' is an exact full-grid sum (sumFluenceExact() in pipeline.ts), chunked to bound")
  lines.push("  readback memory at full-res. GPU-side per-stage (deposit/blurH/blurV) breakdown and a CPU-path")
  lines.push("  comparison are not wired into this report yet; see src/sim/cpuReference.ts + its Vitest suite")
  lines.push("  for the per-stage/CPU-oracle check this currently runs offline instead.")
  lines.push("")

  lines.push("== GPU memory and tiling ==")
  if (!memorySweep || memorySweep.length === 0) {
    lines.push("  not run yet — use the \"Run memory/tiling sweep\" button (COD-252 §4)")
  } else {
    lines.push("  job size sweep at diode pitch (sigma/3 ~= 0.0167mm); estimated bytes = resident deposit/blurA/blurB")
    lines.push("  (RGBA float32) + ramp (RGBA uint8) render targets at that grid size.")
    lines.push("  OK requires: no context loss, every stage's framebuffer COMPLETE, no GL error raised during any")
    lines.push("  stage or the readback, and the exact fluence sum within 1% of the job's deposited energy.")
    for (const step of memorySweep) {
      const status = step.contextLost ? "CONTEXT LOST" : step.success ? "OK" : "FAILED"
      const timing = step.totalMs !== null ? fmtMs(step.totalMs) : "n/a"
      const mb = (step.estimatedBytes / (1024 * 1024)).toFixed(1)
      const ratio = step.fluenceRatio !== null ? step.fluenceRatio.toFixed(4) : "n/a"
      lines.push(`  ${step.sizeMm}mm (grid ${step.gridSize}): ${status}  total=${timing}  est.mem=${mb} MB  fluence ratio=${ratio}`)
      for (const problem of step.problems) lines.push(`    problem: ${problem}`)
      if (step.problems.length === 0 && step.error) lines.push(`    error: ${step.error}`)
      if (step.preexistingGlErrors.length > 0) {
        lines.push(`    (GL errors already pending before this step, not counted: ${step.preexistingGlErrors.join(", ")})`)
      }
    }
    // Largest size such that it and every smaller step passed — a pass above
    // a failure would be suspect, not a usable limit.
    const firstFailure = memorySweep.findIndex((s) => !s.success)
    const passing = firstFailure === -1 ? memorySweep : memorySweep.slice(0, firstFailure)
    const largestOk = passing[passing.length - 1]
    lines.push(
      `  largest untiled size with every step up to it passing: ${largestOk ? `${largestOk.sizeMm}mm (grid ${largestOk.gridSize})` : "none"}`,
    )
  }
  lines.push("")

  lines.push("== CPU fallback ==")
  if (!cpuFallback) {
    lines.push("  not run yet — included automatically when \"Run benchmark\" completes (COD-252 §5)")
  } else {
    lines.push(`  full-res CPU deposit+blur (grid ${cpuFallback.gridSize}x${cpuFallback.gridSize}), run whether or not`)
    lines.push("  EXT_float_blend is present, so the fallback cost is always known:")
    lines.push(`  deposit: ${fmtMs(cpuFallback.depositMs)}`)
    lines.push(`  blur H:  ${fmtMs(cpuFallback.blurHMs)}`)
    lines.push(`  blur V:  ${fmtMs(cpuFallback.blurVMs)}`)
    lines.push(`  total:   ${fmtMs(cpuFallback.totalMs)}`)
  }
  lines.push("")

  lines.push("== IPC ==")
  if (!ipcResults || ipcResults.length === 0) {
    lines.push("  not run yet — use the IPC benchmark panel (COD-252 §6): stream 50k lines from")
    lines.push("  `node scripts/send-gcode.mjs` against each listed mode, and run the status-query RTT check")
    lines.push("  while a stream is in flight.")
  } else {
    for (const r of ipcResults) {
      lines.push(`  ${r.mode} (batch size ${r.batchSize}, port ${r.port}):`)
      lines.push(`    lines: ${r.lineCount}  duration: ${fmtMs(r.durationMs)}  throughput: ${r.linesPerSec.toFixed(0)} lines/sec`)
      lines.push(`    latency p50: ${fmtMs(r.latencyP50Ms)}  p95: ${fmtMs(r.latencyP95Ms)}`)
      if (r.statusRttWhileStreamingMs) {
        const { p50, p95 } = r.statusRttWhileStreamingMs
        lines.push(`    status-query RTT while streaming: p50=${fmtMs(p50)} p95=${fmtMs(p95)}`)
      }
    }
  }
  lines.push("")

  lines.push("== GL errors / warnings ==")
  lines.push(glErrors.length > 0 ? glErrors.join("\n") : "  none observed")

  return lines.join("\n")
}
