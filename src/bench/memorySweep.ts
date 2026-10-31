// GPU memory and tiling sweep (COD-252 §4).
//
// At diode pitch (spotSigmaMm/3, matching the pipeline's deposit spot size),
// a 400mm bed needs ~24000^2 cells — well past most GPUs' MAX_TEXTURE_SIZE,
// so tiling is required in v1. This sweep isn't that tiling implementation;
// it's the measurement that tells us, per machine, how big a single
// untiled job can get before it fails, so the tiling threshold is backed by
// real numbers instead of a guess.

import * as THREE from "three"
import {
  BurnSimPipeline,
  drainGlErrors,
  framebufferStatusName,
  glErrorName,
  stageTimes,
  type PipelineConfig,
} from "../sim/pipeline"
import { discardGpuTimes } from "../sim/gpuTimer"
import { buildTestJob, type TestJob } from "../sim/testJob"

// Matches the spot sigma used elsewhere in the app (App.tsx's executePass);
// diode pitch = sigma/3 per COD-252 §4.
const SPOT_SIGMA_MM = 0.05
export const DIODE_PITCH_MM = SPOT_SIGMA_MM / 3

export interface MemorySweepStep {
  sizeMm: number
  gridSize: number
  success: boolean
  contextLost: boolean
  error: string | null
  totalMs: number | null
  estimatedBytes: number
  // Why a step that didn't throw or lose the context still counts as failed:
  // an incomplete framebuffer or a GL error during any stage, or a fluence
  // total that doesn't match the job's own deposited energy (e.g. zero
  // because a target never allocated and nothing was drawn).
  problems: string[]
  // GL errors already pending before this step ran (from the on-screen
  // render loop or an earlier step) — reported, but not held against it.
  preexistingGlErrors: string[]
  fluenceRatio: number | null // measured / expected; 1.0 = all energy accounted for
}

// Same 1% bound COD-252 §3 sets for fluence conservation. A run whose
// render targets silently failed is off by far more (typically ratio 0).
const FLUENCE_TOLERANCE = 0.01

function expectedFluence(job: TestJob): number {
  let sum = 0
  for (let i = 2; i < job.samples.length; i += 3) sum += job.samples[i]
  return sum
}

// deposit + blurA + blurB are RGBA float32 render targets (see pipeline.ts's
// makeRenderTarget comment on why RGBA, not single-channel); ramp is RGBA
// uint8. The on-screen heightmap samples the ramp target directly rather
// than allocating a separate copy, so these four are the sweep's full
// resident set at any instant.
export function estimateResidentBytes(gridSize: number): number {
  const floatTargetBytes = gridSize * gridSize * 4 * 4
  const byteTargetBytes = gridSize * gridSize * 4
  return 3 * floatTargetBytes + byteTargetBytes
}

// Builds a job scaled to the sweep's job size with a manageable sample
// count: full diode-pitch sampling along a handful of raster lines is
// enough to drive every pipeline stage at the target grid size without the
// multi-million-line sample buffers a true full-density raster at this size
// would need (this sweep measures allocation/rendering limits, not fluence
// accuracy — that's covered by the dedicated full-res/coarse fluence runs).
function buildSweepJob(sizeMm: number, gridSize: number) {
  return buildTestJob({
    sizeMm,
    lineIntervalMm: sizeMm / 50,
    gridPitchMm: sizeMm / gridSize,
    vMaxMmS: 500,
    accelMmS2: 20000,
    power: 1,
    absorption: 1,
  })
}

const DEFAULT_SIZES_MM = [100, 150, 200, 250]

// Steps job size up (COD-252 §4), recording success/failure, context loss,
// and timing at each size, and estimated resident texture memory. Stops
// early if the context is lost, since further GL calls on a lost context
// are meaningless.
//
// "Success" used to mean only "no exception and no context loss", which let
// oversized steps report OK while WebGL quietly recorded INVALID_ENUM /
// INVALID_OPERATION / INVALID_FRAMEBUFFER_OPERATION and drew nothing — the
// first Windows/Iris Xe report showed 250mm (~11 GB est.) as OK, with those
// errors in its global GL-error list and non-monotonic timings. Each step
// now runs with per-stage diagnostics and a fluence sanity check.
export function runMemoryTilingSweep(
  pipeline: BurnSimPipeline,
  renderer: THREE.WebGLRenderer,
  sizesMm: number[] = DEFAULT_SIZES_MM,
): MemorySweepStep[] {
  const steps: MemorySweepStep[] = []
  const gl = renderer.getContext() as WebGL2RenderingContext

  for (const sizeMm of sizesMm) {
    const gridSize = Math.round(sizeMm / DIODE_PITCH_MM)
    const estimatedBytes = estimateResidentBytes(gridSize)

    const stepBase = { sizeMm, gridSize, estimatedBytes, problems: [] as string[], preexistingGlErrors: [] as string[], fluenceRatio: null }
    if (gl.isContextLost()) {
      steps.push({ ...stepBase, success: false, contextLost: true, error: "context already lost; skipped", totalMs: null })
      continue
    }
    const preexistingGlErrors = drainGlErrors(gl).map(glErrorName)

    const config: PipelineConfig = {
      sizeMm,
      gridSize,
      spotSigmaMm: SPOT_SIGMA_MM,
      heatSigmaMm: 0.03,
      rampScale: 600,
      ablationThreshold: 0.6,
    }

    try {
      const job = buildSweepJob(sizeMm, gridSize)
      const result = pipeline.run(job, config, true, { diagnose: true })
      result.renderTarget.dispose()
      discardGpuTimes(gl, stageTimes(result.timings))
      const lost = gl.isContextLost()

      const problems: string[] = []
      for (const d of result.diagnostics ?? []) {
        if (d.framebufferStatus !== null && d.framebufferStatus !== gl.FRAMEBUFFER_COMPLETE) {
          problems.push(`${d.stage}: framebuffer ${framebufferStatusName(d.framebufferStatus)}`)
        }
        if (d.glErrors.length > 0) {
          problems.push(`${d.stage}: GL ${d.glErrors.map(glErrorName).join(", ")}`)
        }
      }
      const expected = expectedFluence(job)
      const fluenceRatio = expected > 0 ? result.totalFluence / expected : null
      if (fluenceRatio === null || !Number.isFinite(fluenceRatio) || Math.abs(fluenceRatio - 1) > FLUENCE_TOLERANCE) {
        problems.push(
          `fluence: measured ${result.totalFluence.toExponential(3)} vs expected ${expected.toExponential(3)}` +
            (fluenceRatio !== null && Number.isFinite(fluenceRatio) ? ` (ratio ${fluenceRatio.toFixed(4)})` : ""),
        )
      }

      steps.push({
        ...stepBase,
        success: !lost && problems.length === 0,
        contextLost: lost,
        error: lost ? "context lost during run" : problems.length > 0 ? problems.join("; ") : null,
        totalMs: result.timings.totalMs,
        problems,
        preexistingGlErrors,
        fluenceRatio,
      })
      if (lost) break
    } catch (err) {
      const lost = gl.isContextLost()
      const errors = drainGlErrors(gl).map(glErrorName)
      steps.push({
        ...stepBase,
        success: false,
        contextLost: lost,
        error: (err instanceof Error ? err.message : String(err)) + (errors.length > 0 ? ` (GL: ${errors.join(", ")})` : ""),
        totalMs: null,
        preexistingGlErrors,
      })
      if (lost) break
    }
  }

  return steps
}
