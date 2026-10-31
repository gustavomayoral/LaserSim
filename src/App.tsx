import { useEffect, useRef, useState } from "react"
import { isTauri } from "@tauri-apps/api/core"
import { buildTestJob, defaultJobParams, type TestJob } from "./sim/testJob"
import { BurnSimPipeline, readGpuCapabilities, type GpuCapabilities } from "./sim/pipeline"
import { HeightmapScene } from "./render/HeightmapScene"
import { buildReportText, type RunSummary, type TimingSeries } from "./bench/report"
import { runMemoryTilingSweep, type MemorySweepStep } from "./bench/memorySweep"
import { runCpuFallbackBenchmark, type CpuFallbackResult } from "./bench/cpuFallback"
import { IPC_PORTS, startStatusRttProbe, summarizeLatencies, type IpcMode, type IpcModeResult } from "./bench/ipcBench"
import "./App.css"

const SIZE_MM = defaultJobParams.sizeMm
const FULL_GRID = 6000
const COARSE_GRID = 500
const GCODE_PORT = 8090

type RunResult = RunSummary

function formatMs(ms: number) {
  return `${ms.toFixed(1)} ms`
}

const MAX_LOG_LINES = 300

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const logRef = useRef<HTMLPreElement>(null)
  const [logLines, setLogLines] = useState<string[]>([])

  // Visible status log (user-requested: the app gave no feedback on what a
  // long-running action — e.g. "Run benchmark" — was actually doing, which
  // made a real failure indistinguishable from "still working"). Every
  // multi-step action below logs its progress here instead of only setting
  // a single status string at the end.
  function log(message: string) {
    const line = `[${new Date().toLocaleTimeString()}] ${message}`
    setLogLines((prev) => {
      const next = [...prev, line]
      return next.length > MAX_LOG_LINES ? next.slice(next.length - MAX_LOG_LINES) : next
    })
    // Mirrored to lasersim-session.log at the project root (src-tauri/lib.rs's
    // append_session_log), so the log is readable without the app's own
    // window — e.g. over SSH while debugging a remote machine. Fire-and-
    // forget: logging must never block or fail the action it's reporting on.
    if (tauriAvailable) {
      import("@tauri-apps/api/core")
        .then(({ invoke }) => invoke("append_session_log", { line }))
        .catch(() => {})
    }
  }

  useEffect(() => {
    const el = logRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [logLines])

  const sceneRef = useRef<HeightmapScene | null>(null)
  const pipelineRef = useRef<BurnSimPipeline | null>(null)
  const jobsRef = useRef<{ full: TestJob; coarse: TestJob } | null>(null)

  const [caps, setCaps] = useState<GpuCapabilities | null>(null)
  const [initError, setInitError] = useState<string | null>(null)
  const [fullResult, setFullResult] = useState<RunResult | null>(null)
  const [coarseResult, setCoarseResult] = useState<RunResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [gcodeCount, setGcodeCount] = useState(0)
  const [gcodeRate, setGcodeRate] = useState<number | null>(null)
  const [gcodeLatencyMs, setGcodeLatencyMs] = useState<number | null>(null)
  const tauriAvailable = isTauri()

  const [benchBusy, setBenchBusy] = useState(false)
  const [benchStatus, setBenchStatus] = useState<string | null>(null)
  // Editable save path (text input, not a native dialog — tauri-plugin-
  // dialog's save() hung indefinitely on at least one real Windows/WebView2
  // machine, no dialog ever appeared even after checking the taskbar, so a
  // plain field the user can read/edit themselves replaced it entirely).
  const [savePath, setSavePath] = useState("")
  const lastReportRef = useRef<string | null>(null)

  const [memorySweep, setMemorySweep] = useState<MemorySweepStep[] | null>(null)
  const [memorySweepBusy, setMemorySweepBusy] = useState(false)
  const [cpuFallback, setCpuFallback] = useState<CpuFallbackResult | null>(null)
  const [coarseUnsyncedMs, setCoarseUnsyncedMs] = useState<number | null>(null)

  const [ipcResults, setIpcResults] = useState<IpcModeResult[]>([])
  const [ipcBusyMode, setIpcBusyMode] = useState<IpcMode | null>(null)
  const ipcSessionRef = useRef<{
    mode: IpcMode
    batchSize: number
    firstMs: number | null
    lastMs: number | null
    lineCount: number
    latenciesMs: number[]
    rttProbe: ReturnType<typeof startStatusRttProbe> | null
  } | null>(null)

  useEffect(() => {
    if (!canvasRef.current) return
    // Uncaught here would crash the whole mount (React unmounts to a blank
    // page on an error thrown from an effect, with nothing in the UI to say
    // why) — this runs scene/context creation and a full warmup pass on a
    // GPU/driver combination that may behave differently than the one this
    // was developed against, so failure here must be visible, not fatal.
    let scene: HeightmapScene | null = null
    log("Initializing GPU scene...")
    try {
      scene = new HeightmapScene(canvasRef.current, SIZE_MM)
      sceneRef.current = scene
      const pipeline = new BurnSimPipeline(scene.renderer)
      pipelineRef.current = pipeline
      const gpuCaps = readGpuCapabilities(scene.renderer)
      setCaps(gpuCaps)
      log(`GPU: ${gpuCaps.renderer ?? "unknown"} (WebGL2: ${gpuCaps.isWebGL2})`)
      log("Warming up shaders...")
      pipeline.warmup()
      log("Ready.")
    } catch (err) {
      const message = `Startup failed: ${err instanceof Error ? err.message : String(err)}`
      log(message)
      setInitError(message)
    }
    if (!scene) return

    const resize = () => {
      const el = canvasRef.current
      if (!el) return
      scene!.resize(el.clientWidth, el.clientHeight)
    }
    resize()
    window.addEventListener("resize", resize)
    return () => {
      window.removeEventListener("resize", resize)
      scene!.dispose()
    }
  }, [])

  useEffect(() => {
    if (!tauriAvailable) return
    let firstMs: number | null = null
    let lastMs: number | null = null
    let latencies: number[] = []
    let rttProbe: ReturnType<typeof startStatusRttProbe> | null = null
    let unlistenLine: (() => void) | undefined
    let unlistenDone: (() => void) | undefined
    ;(async () => {
      const { listen } = await import("@tauri-apps/api/event")
      unlistenLine = await listen<{ seq: number; line: string; rust_ms: number }>("gcode-line", (event) => {
        const nowMs = Date.now()
        if (firstMs === null) {
          firstMs = nowMs
          latencies = []
          // Measures invoke() RTT concurrently with this per-line-event
          // stream, for the COD-252 §6 "status query while a stream is in
          // flight" check.
          rttProbe = startStatusRttProbe()
        }
        lastMs = nowMs
        setGcodeCount(event.payload.seq)
        // Rough one-way latency: Rust wall clock vs JS wall clock, both on the
        // same machine. Not a precise measurement — see COD-251 notes.
        const latencyMs = nowMs - event.payload.rust_ms
        latencies.push(latencyMs)
        setGcodeLatencyMs(latencyMs)
      })
      unlistenDone = await listen<{ seq: number }>("gcode-done", (event) => {
        ;(async () => {
          const probe = rttProbe
          rttProbe = null
          const statusRttWhileStreamingMs = probe ? await probe.stop() : null
          if (firstMs !== null && lastMs !== null && lastMs > firstMs) {
            const durationMs = lastMs - firstMs
            const linesPerSec = (event.payload.seq / durationMs) * 1000
            setGcodeRate(linesPerSec)
            const { p50, p95 } = summarizeLatencies(latencies)
            const result: IpcModeResult = {
              mode: "per-line-events",
              batchSize: 1,
              port: IPC_PORTS["per-line-events"],
              lineCount: event.payload.seq,
              durationMs,
              linesPerSec,
              latencyP50Ms: p50,
              latencyP95Ms: p95,
              statusRttWhileStreamingMs,
            }
            setIpcResults((prev) => [...prev.filter((r) => r.mode !== "per-line-events"), result])
          }
          firstMs = null
          lastMs = null
        })()
      })
    })()
    return () => {
      unlistenLine?.()
      unlistenDone?.()
      rttProbe?.stop()
    }
  }, [tauriAvailable])

  useEffect(() => {
    if (!tauriAvailable) return
    ;(async () => {
      try {
        const { invoke } = await import("@tauri-apps/api/core")
        const defaultPath = await invoke<string>("default_benchmark_report_path")
        setSavePath(defaultPath)
      } catch (err) {
        log(`Could not compute a default save path: ${err instanceof Error ? err.message : String(err)}`)
      }
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tauriAvailable])

  async function saveReport() {
    const report = lastReportRef.current
    if (!report) {
      log("Save report: nothing to save yet — run the benchmark first.")
      return
    }
    if (!savePath) {
      log("Save report: path is empty.")
      return
    }
    log(`Writing report to ${savePath}...`)
    try {
      const { invoke } = await import("@tauri-apps/api/core")
      const path = await invoke<string>("write_benchmark_report", { path: savePath, report })
      log(`Report written to ${path}`)
      setBenchStatus(`Report written to ${path}`)
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      log(`Failed to write report: ${message}`)
      setBenchStatus(`Failed to write report: ${message}`)
    }
  }

  const IPC_BATCH_SIZE = 256

  // Shared finalize step for the batched-events and channel IPC modes: both
  // accumulate lines/latencies into ipcSessionRef and call this once their
  // stream-end signal arrives.
  async function finalizeIpcSession() {
    const session = ipcSessionRef.current
    ipcSessionRef.current = null
    setIpcBusyMode(null)
    if (!session || session.firstMs === null || session.lastMs === null || session.lastMs <= session.firstMs) return
    const statusRttWhileStreamingMs = session.rttProbe ? await session.rttProbe.stop() : null
    const durationMs = session.lastMs - session.firstMs
    const linesPerSec = (session.lineCount / durationMs) * 1000
    const { p50, p95 } = summarizeLatencies(session.latenciesMs)
    const result: IpcModeResult = {
      mode: session.mode,
      batchSize: session.batchSize,
      port: IPC_PORTS[session.mode],
      lineCount: session.lineCount,
      durationMs,
      linesPerSec,
      latencyP50Ms: p50,
      latencyP95Ms: p95,
      statusRttWhileStreamingMs,
    }
    setIpcResults((prev) => [...prev.filter((r) => r.mode !== session.mode), result])
    log(`IPC (${session.mode}): ${session.lineCount} lines, ${linesPerSec.toFixed(0)} lines/sec, p50=${p50.toFixed(2)}ms p95=${p95.toFixed(2)}ms`)
  }

  async function startBatchedIpcListener() {
    if (!tauriAvailable || ipcBusyMode) return
    setError(null)
    setIpcBusyMode("batched-events")
    const port = IPC_PORTS["batched-events"]
    log(`IPC: listening for batched-events run on port ${port}...`)
    try {
      const { invoke } = await import("@tauri-apps/api/core")
      const { listen } = await import("@tauri-apps/api/event")
      // Session and listeners go in before the bind, so a sender that
      // connects the instant the port opens can't race past them.
      ipcSessionRef.current = {
        mode: "batched-events",
        batchSize: IPC_BATCH_SIZE,
        firstMs: null,
        lastMs: null,
        lineCount: 0,
        latenciesMs: [],
        rttProbe: null,
      }
      const unlistenBatch = await listen<{ seq_start: number; count: number; rust_ms: number }>("gcode-batch", (event) => {
        const session = ipcSessionRef.current
        if (!session) return
        const nowMs = Date.now()
        if (session.firstMs === null) {
          session.firstMs = nowMs
          session.rttProbe = startStatusRttProbe()
        }
        session.lastMs = nowMs
        session.lineCount += event.payload.count
        session.latenciesMs.push(nowMs - event.payload.rust_ms)
      })
      const unlistenDone = await listen<{ total: number }>("gcode-batch-done", () => {
        unlistenBatch()
        unlistenDone()
        void finalizeIpcSession()
      })
      try {
        await invoke("start_gcode_batched", { port, batchSize: IPC_BATCH_SIZE })
      } catch (err) {
        unlistenBatch()
        unlistenDone()
        throw err
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      log(`IPC batched-events listener FAILED: ${message}`)
      ipcSessionRef.current = null
      setIpcBusyMode(null)
      setError(message)
    }
  }

  async function startChannelIpcListener() {
    if (!tauriAvailable || ipcBusyMode) return
    setError(null)
    setIpcBusyMode("channel")
    const port = IPC_PORTS.channel
    log(`IPC: listening for channel run on port ${port}...`)
    try {
      const { invoke, Channel } = await import("@tauri-apps/api/core")
      ipcSessionRef.current = {
        mode: "channel",
        batchSize: IPC_BATCH_SIZE,
        firstMs: null,
        lastMs: null,
        lineCount: 0,
        latenciesMs: [],
        rttProbe: null,
      }
      const channel = new Channel<{ seq_start: number; count: number; rust_ms: number }>()
      channel.onmessage = (payload) => {
        const session = ipcSessionRef.current
        if (!session) return
        if (payload.count === 0) {
          // Zero-count sentinel (see ipc_bench.rs's start_gcode_channel):
          // the stream ended.
          void finalizeIpcSession()
          return
        }
        const nowMs = Date.now()
        if (session.firstMs === null) {
          session.firstMs = nowMs
          session.rttProbe = startStatusRttProbe()
        }
        session.lastMs = nowMs
        session.lineCount += payload.count
        session.latenciesMs.push(nowMs - payload.rust_ms)
      }
      await invoke("start_gcode_channel", { port, batchSize: IPC_BATCH_SIZE, channel })
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      log(`IPC channel listener FAILED: ${message}`)
      ipcSessionRef.current = null
      setIpcBusyMode(null)
      setError(message)
    }
  }

  function getJobs() {
    if (!jobsRef.current) {
      jobsRef.current = {
        full: buildTestJob({ ...defaultJobParams, gridPitchMm: SIZE_MM / FULL_GRID }),
        coarse: buildTestJob({ ...defaultJobParams, gridPitchMm: SIZE_MM / COARSE_GRID }),
      }
    }
    return jobsRef.current
  }

  // Runs one pass synchronously and returns its summary; callers decide when
  // to let the UI paint "busy" first. Throws on pipeline error.
  function executePass(kind: "full" | "coarse"): RunResult {
    const pipeline = pipelineRef.current
    const scene = sceneRef.current
    if (!pipeline || !scene) throw new Error("pipeline not initialized")
    const gridSize = kind === "full" ? FULL_GRID : COARSE_GRID
    const maxGrid = caps?.maxTextureSize ?? gridSize
    const clampedGrid = Math.min(gridSize, maxGrid)
    const job = kind === "full" ? getJobs().full : getJobs().coarse
    const result = pipeline.run(job, {
      sizeMm: SIZE_MM,
      gridSize: clampedGrid,
      spotSigmaMm: 0.05,
      heatSigmaMm: 0.03,
      // Empirically-tuned visualization "exposure" so cruise-speed fluence
      // (raw units ~1e-3, since energy = power*dt*absorption with dt in
      // seconds) lands mid-ramp and only the slowed line-ends go dark.
      // Not a physical calibration — see "Out of scope" in COD-251.
      rampScale: 600,
      ablationThreshold: 0.6,
    })
    scene.setTexture(result.renderTarget, result.gridSize)
    return { gridSize: result.gridSize, timings: result.timings, totalFluence: result.totalFluence }
  }

  function runPass(kind: "full" | "coarse") {
    setBusy(true)
    setError(null)
    log(`Running ${kind} pass...`)
    // Let the UI paint "busy" before the (synchronous, blocking) GPU work.
    requestAnimationFrame(() => {
      try {
        const summary = executePass(kind)
        if (kind === "full") setFullResult(summary)
        else setCoarseResult(summary)
        log(`${kind} pass done: total ${summary.timings.totalMs.toFixed(1)} ms`)
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        log(`${kind} pass FAILED: ${message}`)
        setError(message)
      } finally {
        setBusy(false)
      }
    })
  }

  // Builds a full-grid job/config pair matching executePass("full"), for the
  // CPU-fallback timing (COD-252 §5), which needs the job and config rather
  // than just the GPU-side RunResult.
  function fullConfig() {
    return {
      sizeMm: SIZE_MM,
      gridSize: Math.min(FULL_GRID, caps?.maxTextureSize ?? FULL_GRID),
      spotSigmaMm: 0.05,
      heatSigmaMm: 0.03,
    }
  }

  function runBenchmark() {
    const scene = sceneRef.current
    const pipeline = pipelineRef.current
    if (!scene || !pipeline || !caps) {
      log("Run benchmark: ignored — scene/pipeline/capabilities aren't ready yet (wait for the GPU capabilities panel to finish probing).")
      return
    }
    log("Run benchmark: starting...")
    setBenchBusy(true)
    setBenchStatus(null)
    setError(null)
    requestAnimationFrame(() => {
      try {
        log("Running full-res pass...")
        const full = executePass("full")
        log(`Full-res done: total ${full.timings.totalMs.toFixed(1)} ms`)
        log("Running coarse-scrub pass...")
        const coarse = executePass("coarse")
        log(`Coarse-scrub done: total ${coarse.timings.totalMs.toFixed(1)} ms`)
        setFullResult(full)
        setCoarseResult(coarse)

        // Warm/cold timing series (COD-252 §1/§2): separate from the
        // single display runs above so the heightmap preview still reflects
        // the last real run rather than a disposed timing-only one.
        log("Running full-res timing series (1 cold + 10 warm runs)...")
        const fullSeries: TimingSeries = pipeline.runSeries(getJobs().full, {
          sizeMm: SIZE_MM,
          gridSize: full.gridSize,
          spotSigmaMm: 0.05,
          heatSigmaMm: 0.03,
          rampScale: 600,
          ablationThreshold: 0.6,
        })
        log("Running coarse-scrub timing series (1 cold + 10 warm runs)...")
        const coarseSeries: TimingSeries = pipeline.runSeries(getJobs().coarse, {
          sizeMm: SIZE_MM,
          gridSize: coarse.gridSize,
          spotSigmaMm: 0.05,
          heatSigmaMm: 0.03,
          rampScale: 600,
          ablationThreshold: 0.6,
        })

        log("Running CPU fallback deposit+blur on the full-res job (this can take several seconds)...")
        const fc = fullConfig()
        const cpuFallbackResult = runCpuFallbackBenchmark(getJobs().full, SIZE_MM, fc.gridSize, fc.spotSigmaMm, fc.heatSigmaMm)
        setCpuFallback(cpuFallbackResult)
        log(`CPU fallback done: total ${cpuFallbackResult.totalMs.toFixed(1)} ms`)

        // Unsynced estimate of what real interactive scrubbing would cost
        // (COD-252 §2 discussion): the synced numbers above are correct for
        // the benchmark but pay a per-stage gl.finish()/readPixels cost real
        // interactive use wouldn't need. Disposed immediately; only its
        // timing matters here.
        log("Running unsynced coarse-scrub estimate...")
        const coarseUnsynced = pipeline.run(
          getJobs().coarse,
          { sizeMm: SIZE_MM, gridSize: coarse.gridSize, spotSigmaMm: 0.05, heatSigmaMm: 0.03, rampScale: 600, ablationThreshold: 0.6 },
          false,
        )
        coarseUnsynced.renderTarget.dispose()
        setCoarseUnsyncedMs(coarseUnsynced.timings.totalMs)

        const gl = scene.renderer.getContext()
        const glErrors: string[] = []
        let glErr = gl.getError()
        while (glErr !== gl.NO_ERROR) {
          glErrors.push(`GL error code ${glErr}`)
          glErr = gl.getError()
        }

        function expectedFluenceFor(samples: Float32Array) {
          let sum = 0
          for (let i = 2; i < samples.length; i += 3) sum += samples[i]
          return sum
        }
        const expectedFluenceFull = expectedFluenceFor(getJobs().full.samples)
        const expectedFluenceCoarse = expectedFluenceFor(getJobs().coarse.samples)

        const report = buildReportText({
          caps,
          full,
          coarse,
          expectedFluenceFull,
          expectedFluenceCoarse,
          glErrors,
          fullSeries,
          coarseSeries,
          memorySweep,
          cpuFallback: cpuFallbackResult,
          ipcResults: ipcResults.length > 0 ? ipcResults : null,
          coarseUnsyncedTotalMs: coarseUnsynced.timings.totalMs,
        })

        lastReportRef.current = report

        if (tauriAvailable) {
          log("Report built. Writing to the save path shown below (edit it and click \"Save report\" to resave elsewhere)...")
          void saveReport().finally(() => setBenchBusy(false))
        } else {
          console.log(report)
          log("Not running inside Tauri — report logged to the browser console instead of written to disk.")
          setBenchStatus("Not running inside Tauri — report logged to the browser console instead of written to disk.")
          setBenchBusy(false)
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        log(`Run benchmark FAILED: ${message}`)
        setError(message)
        setBenchBusy(false)
      }
    })
  }

  function runMemorySweep() {
    const scene = sceneRef.current
    const pipeline = pipelineRef.current
    if (!scene || !pipeline) {
      log("Run memory/tiling sweep: ignored — scene/pipeline not ready yet.")
      return
    }
    log("Run memory/tiling sweep: starting (100/150/200/250mm)...")
    setMemorySweepBusy(true)
    setError(null)
    requestAnimationFrame(() => {
      try {
        const steps = runMemoryTilingSweep(pipeline, scene.renderer)
        setMemorySweep(steps)
        for (const step of steps) {
          const status = step.contextLost ? "CONTEXT LOST" : step.success ? "OK" : "FAILED"
          log(`  ${step.sizeMm}mm (grid ${step.gridSize}): ${status}${step.error ? ` — ${step.error}` : ""}`)
        }
        log("Memory/tiling sweep done.")
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        log(`Memory/tiling sweep FAILED: ${message}`)
        setError(message)
      } finally {
        setMemorySweepBusy(false)
      }
    })
  }

  const fluenceRatio =
    fullResult && coarseResult && coarseResult.totalFluence > 0
      ? fullResult.totalFluence / coarseResult.totalFluence
      : null

  return (
    <main className="container">
      <h1>LaserSim — COD-251 spike</h1>
      <p className="subtitle">
        Tauri + WebGL2 burn-sim performance test: deposit (bilinear points, additive) → separable Gaussian blur →
        material ramp → three.js heightmap.
      </p>

      <section className="panel">
        <h2>Status log</h2>
        <pre ref={logRef} className="status-log">
          {logLines.length > 0 ? logLines.join("\n") : "(idle — nothing has run yet this session)"}
        </pre>
      </section>

      {initError && (
        <section className="panel">
          <p className="error">{initError}</p>
        </section>
      )}

      <section className="panel">
        <h2>GPU capabilities</h2>
        {caps ? (
          <ul>
            <li>WebGL2: {caps.isWebGL2 ? "yes" : "NO — pipeline requires it"}</li>
            <li>Max texture size: {caps.maxTextureSize}</li>
            <li>
              EXT_color_buffer_float: {caps.colorBufferFloat ? "available" : "MISSING — float render targets unsupported"}
            </li>
            <li>
              EXT_float_blend: {caps.floatBlend ? "available" : "MISSING — additive deposit blending is unsupported here"}
            </li>
            <li>Max renderbuffer size: {caps.maxRenderbufferSize}</li>
            <li>OES_texture_float_linear: {caps.textureFloatLinear ? "available" : "missing"}</li>
            <li>EXT_disjoint_timer_query_webgl2: {caps.disjointTimerQuery ? "available" : "missing"}</li>
            <li>GL renderer: {caps.renderer ?? "unavailable"}</li>
            <li>GL vendor: {caps.vendor ?? "unavailable"}</li>
          </ul>
        ) : (
          <p>probing…</p>
        )}
      </section>

      <section className="panel">
        <h2>Benchmark report (COD-252)</h2>
        <div className="row">
          <button disabled={benchBusy || busy} onClick={runBenchmark}>
            Run benchmark
          </button>
        </div>
        <p className="note">
          Runs the full-res and coarse passes (single run, plus a 1-cold + 10-warm timing series per COD-252 §2),
          times the CPU fallback deposit+blur on the full-res job (§5), and writes a plain-text report (environment,
          capabilities, timings, fluence conservation, memory/tiling, CPU fallback, IPC). Memory/tiling (§4) and IPC
          (§6) results are only included if you've run those panels below first in this session.
        </p>
        {tauriAvailable && (
          <div className="row save-path-row">
            <input
              type="text"
              value={savePath}
              onChange={(e) => setSavePath(e.target.value)}
              placeholder="Full path to save the report to"
              className="save-path-input"
            />
            <button onClick={() => void saveReport()}>Save report</button>
          </div>
        )}
        {benchStatus && <p className="note">{benchStatus}</p>}
        {cpuFallback && (
          <p className="note">
            CPU fallback (grid {cpuFallback.gridSize}²): deposit {formatMs(cpuFallback.depositMs)}, blur H{" "}
            {formatMs(cpuFallback.blurHMs)}, blur V {formatMs(cpuFallback.blurVMs)}, total{" "}
            {formatMs(cpuFallback.totalMs)}.
          </p>
        )}
        {coarseUnsyncedMs !== null && (
          <p className="note">
            Coarse scrub, unsynced (interactive-use estimate, not the GPU-complete number above): {formatMs(coarseUnsyncedMs)}{" "}
            — {coarseUnsyncedMs < 50 ? "PASS" : "FAIL"} against the &lt;50ms criterion. The synced coarse-scrub total pays a
            per-stage gl.finish()/readPixels cost real interactive use wouldn't need.
          </p>
        )}
      </section>

      <section className="panel">
        <h2>Pipeline runs</h2>
        <div className="row">
          <button disabled={busy} onClick={() => runPass("full")}>
            Run full-res pass ({FULL_GRID}²)
          </button>
          <button disabled={busy} onClick={() => runPass("coarse")}>
            Run coarse scrub pass ({COARSE_GRID}²)
          </button>
        </div>
        {error && <p className="error">Error: {error}</p>}
        <table>
          <thead>
            <tr>
              <th></th>
              <th>deposit</th>
              <th>blur H</th>
              <th>blur V</th>
              <th>ramp</th>
              <th>total</th>
              <th>pass criterion</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Full-res (&lt; 2 s)</td>
              <td>{fullResult ? formatMs(fullResult.timings.deposit.wallMs) : "—"}</td>
              <td>{fullResult ? formatMs(fullResult.timings.blurH.wallMs) : "—"}</td>
              <td>{fullResult ? formatMs(fullResult.timings.blurV.wallMs) : "—"}</td>
              <td>{fullResult ? formatMs(fullResult.timings.ramp.wallMs) : "—"}</td>
              <td>{fullResult ? formatMs(fullResult.timings.totalMs) : "—"}</td>
              <td>{fullResult ? (fullResult.timings.totalMs < 2000 ? "PASS" : "FAIL") : "—"}</td>
            </tr>
            <tr>
              <td>Coarse scrub (&lt; 50 ms)</td>
              <td>{coarseResult ? formatMs(coarseResult.timings.deposit.wallMs) : "—"}</td>
              <td>{coarseResult ? formatMs(coarseResult.timings.blurH.wallMs) : "—"}</td>
              <td>{coarseResult ? formatMs(coarseResult.timings.blurV.wallMs) : "—"}</td>
              <td>{coarseResult ? formatMs(coarseResult.timings.ramp.wallMs) : "—"}</td>
              <td>{coarseResult ? formatMs(coarseResult.timings.totalMs) : "—"}</td>
              <td>{coarseResult ? (coarseResult.timings.totalMs < 50 ? "PASS" : "FAIL") : "—"}</td>
            </tr>
          </tbody>
        </table>
        <p className="note">
          Total fluence (exact full-grid sum): full-res {fullResult?.totalFluence.toExponential(3) ?? "—"}, coarse{" "}
          {coarseResult?.totalFluence.toExponential(3) ?? "—"}
          {fluenceRatio !== null && ` — ratio ${fluenceRatio.toFixed(3)} (1.0 = perfectly consistent)`}. See "Run benchmark" below for
          the expected-vs-measured conservation check against each job's own sample energy.
        </p>
      </section>

      <section className="panel">
        <h2>GPU memory and tiling sweep (COD-252 §4)</h2>
        <div className="row">
          <button disabled={memorySweepBusy || busy || benchBusy} onClick={runMemorySweep}>
            Run memory/tiling sweep
          </button>
        </div>
        <p className="note">
          Steps job size (100/150/200/250mm square) at diode pitch (spotSigma/3 ≈ 0.0167mm), recording success,
          context loss, and timing at each size, plus estimated resident texture memory. Included in the next "Run
          benchmark" report.
        </p>
        {memorySweep && (
          <table>
            <thead>
              <tr>
                <th>size</th>
                <th>grid</th>
                <th>status</th>
                <th>total</th>
                <th>est. memory</th>
              </tr>
            </thead>
            <tbody>
              {memorySweep.map((step) => (
                <tr key={step.sizeMm}>
                  <td>{step.sizeMm}mm</td>
                  <td>{step.gridSize}</td>
                  <td>{step.contextLost ? "CONTEXT LOST" : step.success ? "OK" : "FAILED"}</td>
                  <td>{step.totalMs !== null ? formatMs(step.totalMs) : "—"}</td>
                  <td>{(step.estimatedBytes / (1024 * 1024)).toFixed(1)} MB</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="panel">
        <h2>IPC benchmark (COD-252 §6)</h2>
        {tauriAvailable ? (
          <>
            <p>
              Three transports, same <code>node scripts/send-gcode.mjs [lineCount] [port]</code> sender: per-line
              global events (COD-251 baseline, always listening on port {GCODE_PORT}), events batched{" "}
              {IPC_BATCH_SIZE} lines/message (port {IPC_PORTS["batched-events"]}), and a Tauri v2 Channel batched the
              same way (port {IPC_PORTS.channel}). Each run also measures <code>invoke("ping")</code> round-trip
              latency concurrently, for the "status query while a stream is in flight" check.
            </p>
            <div className="row">
              <button disabled={ipcBusyMode !== null} onClick={startBatchedIpcListener}>
                Listen for batched-events run (port {IPC_PORTS["batched-events"]})
              </button>
              <button disabled={ipcBusyMode !== null} onClick={startChannelIpcListener}>
                Listen for channel run (port {IPC_PORTS.channel})
              </button>
            </div>
            {ipcBusyMode && (
              <p className="note">
                Listening ({ipcBusyMode}) — run{" "}
                <code>
                  node scripts/send-gcode.mjs 50000 {IPC_PORTS[ipcBusyMode]}
                </code>{" "}
                now.
              </p>
            )}
            <ul>
              <li>Baseline lines received (live): {gcodeCount}</li>
              <li>Baseline receive rate (live): {gcodeRate !== null ? `${gcodeRate.toFixed(0)} lines/sec` : "—"}</li>
              <li>
                Baseline approx. one-way latency (live): {gcodeLatencyMs !== null ? `${gcodeLatencyMs.toFixed(1)} ms` : "—"}
              </li>
            </ul>
            {ipcResults.length > 0 && (
              <table>
                <thead>
                  <tr>
                    <th>mode</th>
                    <th>lines</th>
                    <th>throughput</th>
                    <th>latency p50</th>
                    <th>latency p95</th>
                    <th>status RTT p50/p95 while streaming</th>
                  </tr>
                </thead>
                <tbody>
                  {ipcResults.map((r) => (
                    <tr key={r.mode}>
                      <td>{r.mode}</td>
                      <td>{r.lineCount}</td>
                      <td>{r.linesPerSec.toFixed(0)} lines/sec</td>
                      <td>{formatMs(r.latencyP50Ms)}</td>
                      <td>{formatMs(r.latencyP95Ms)}</td>
                      <td>
                        {r.statusRttWhileStreamingMs
                          ? `${formatMs(r.statusRttWhileStreamingMs.p50)} / ${formatMs(r.statusRttWhileStreamingMs.p95)}`
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </>
        ) : (
          <p>
            Not running inside the Tauri shell (this is the plain Vite dev page) — the native TCP listeners aren't
            reachable here. Run <code>npm run tauri dev</code> to test this section.
          </p>
        )}
      </section>

      <section className="panel canvas-panel">
        <h2>Heightmap preview (drag to orbit)</h2>
        <canvas ref={canvasRef} className="heightmap-canvas" />
      </section>
    </main>
  )
}

export default App
