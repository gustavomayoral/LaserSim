import { useEffect, useRef, useState } from "react"
import { isTauri } from "@tauri-apps/api/core"
import { buildTestJob, defaultJobParams, type TestJob } from "./sim/testJob"
import { BurnSimPipeline, readGpuCapabilities, type GpuCapabilities, type PassTimings } from "./sim/pipeline"
import { HeightmapScene } from "./render/HeightmapScene"
import "./App.css"

const SIZE_MM = defaultJobParams.sizeMm
const FULL_GRID = 6000
const COARSE_GRID = 500
const GCODE_PORT = 8090

interface RunResult {
  gridSize: number
  timings: PassTimings
  totalFluence: number
}

function formatMs(ms: number) {
  return `${ms.toFixed(1)} ms`
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sceneRef = useRef<HeightmapScene | null>(null)
  const pipelineRef = useRef<BurnSimPipeline | null>(null)
  const jobsRef = useRef<{ full: TestJob; coarse: TestJob } | null>(null)

  const [caps, setCaps] = useState<GpuCapabilities | null>(null)
  const [fullResult, setFullResult] = useState<RunResult | null>(null)
  const [coarseResult, setCoarseResult] = useState<RunResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [gcodeCount, setGcodeCount] = useState(0)
  const [gcodeRate, setGcodeRate] = useState<number | null>(null)
  const [gcodeLatencyMs, setGcodeLatencyMs] = useState<number | null>(null)
  const tauriAvailable = isTauri()

  useEffect(() => {
    if (!canvasRef.current) return
    const scene = new HeightmapScene(canvasRef.current, SIZE_MM)
    sceneRef.current = scene
    pipelineRef.current = new BurnSimPipeline(scene.renderer)
    setCaps(readGpuCapabilities(scene.renderer))

    const resize = () => {
      const el = canvasRef.current
      if (!el) return
      scene.resize(el.clientWidth, el.clientHeight)
    }
    resize()
    window.addEventListener("resize", resize)
    return () => {
      window.removeEventListener("resize", resize)
      scene.dispose()
    }
  }, [])

  useEffect(() => {
    if (!tauriAvailable) return
    let firstMs: number | null = null
    let lastMs: number | null = null
    let unlistenLine: (() => void) | undefined
    let unlistenDone: (() => void) | undefined
    ;(async () => {
      const { listen } = await import("@tauri-apps/api/event")
      unlistenLine = await listen<{ seq: number; line: string; rust_ms: number }>("gcode-line", (event) => {
        const nowMs = Date.now()
        if (firstMs === null) firstMs = nowMs
        lastMs = nowMs
        setGcodeCount(event.payload.seq)
        // Rough one-way latency: Rust wall clock vs JS wall clock, both on the
        // same machine. Not a precise measurement — see COD-251 notes.
        setGcodeLatencyMs(nowMs - event.payload.rust_ms)
      })
      unlistenDone = await listen<{ seq: number }>("gcode-done", (event) => {
        if (firstMs !== null && lastMs !== null && lastMs > firstMs) {
          setGcodeRate((event.payload.seq / (lastMs - firstMs)) * 1000)
        }
        firstMs = null
        lastMs = null
      })
    })()
    return () => {
      unlistenLine?.()
      unlistenDone?.()
    }
  }, [tauriAvailable])

  function getJobs() {
    if (!jobsRef.current) {
      jobsRef.current = {
        full: buildTestJob({ ...defaultJobParams, gridPitchMm: SIZE_MM / FULL_GRID }),
        coarse: buildTestJob({ ...defaultJobParams, gridPitchMm: SIZE_MM / COARSE_GRID }),
      }
    }
    return jobsRef.current
  }

  function runPass(kind: "full" | "coarse") {
    const pipeline = pipelineRef.current
    const scene = sceneRef.current
    if (!pipeline || !scene) return
    setBusy(true)
    setError(null)
    // Let the UI paint "busy" before the (synchronous, blocking) GPU work.
    requestAnimationFrame(() => {
      try {
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
        const summary: RunResult = { gridSize: result.gridSize, timings: result.timings, totalFluence: result.totalFluence }
        if (kind === "full") setFullResult(summary)
        else setCoarseResult(summary)
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
      } finally {
        setBusy(false)
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
          </ul>
        ) : (
          <p>probing…</p>
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
              <th>blur</th>
              <th>ramp</th>
              <th>total</th>
              <th>pass criterion</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Full-res (&lt; 2 s)</td>
              <td>{fullResult ? formatMs(fullResult.timings.depositMs) : "—"}</td>
              <td>{fullResult ? formatMs(fullResult.timings.blurMs) : "—"}</td>
              <td>{fullResult ? formatMs(fullResult.timings.rampMs) : "—"}</td>
              <td>{fullResult ? formatMs(fullResult.timings.totalMs) : "—"}</td>
              <td>{fullResult ? (fullResult.timings.totalMs < 2000 ? "PASS" : "FAIL") : "—"}</td>
            </tr>
            <tr>
              <td>Coarse scrub (&lt; 50 ms)</td>
              <td>{coarseResult ? formatMs(coarseResult.timings.depositMs) : "—"}</td>
              <td>{coarseResult ? formatMs(coarseResult.timings.blurMs) : "—"}</td>
              <td>{coarseResult ? formatMs(coarseResult.timings.rampMs) : "—"}</td>
              <td>{coarseResult ? formatMs(coarseResult.timings.totalMs) : "—"}</td>
              <td>{coarseResult ? (coarseResult.timings.totalMs < 50 ? "PASS" : "FAIL") : "—"}</td>
            </tr>
          </tbody>
        </table>
        <p className="note">
          Total fluence (sampled 20mm² corner, scaled): full-res {fullResult?.totalFluence.toExponential(3) ?? "—"}, coarse{" "}
          {coarseResult?.totalFluence.toExponential(3) ?? "—"}
          {fluenceRatio !== null && ` — ratio ${fluenceRatio.toFixed(3)} (1.0 = perfectly consistent)`}.
        </p>
      </section>

      <section className="panel">
        <h2>GRBL TCP emulator IPC check</h2>
        {tauriAvailable ? (
          <>
            <p>
              Listening on 127.0.0.1:{GCODE_PORT}. Run{" "}
              <code>node scripts/send-gcode.mjs [lineCount] [port]</code> from a terminal while this window is open.
            </p>
            <ul>
              <li>Lines received: {gcodeCount}</li>
              <li>Receive rate: {gcodeRate !== null ? `${gcodeRate.toFixed(0)} lines/sec` : "—"}</li>
              <li>
                Approx. one-way latency (Rust wall clock → JS receipt): {gcodeLatencyMs !== null ? `${gcodeLatencyMs.toFixed(1)} ms` : "—"}{" "}
                — coarse, same-machine clocks, see COD-251 notes.
              </li>
            </ul>
          </>
        ) : (
          <p>
            Not running inside the Tauri shell (this is the plain Vite dev page) — the native TCP listener isn't
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
