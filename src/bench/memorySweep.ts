// GPU memory and tiling sweep (COD-252 §4).
//
// At diode pitch (spotSigmaMm/3, matching the pipeline's deposit spot size),
// a 400mm bed needs ~24000^2 cells — well past most GPUs' MAX_TEXTURE_SIZE,
// so tiling is required in v1. This sweep isn't that tiling implementation;
// it's the measurement that tells us, per machine, how big a single
// untiled job can get before it fails, so the tiling threshold is backed by
// real numbers instead of a guess.

import * as THREE from "three"
import { BurnSimPipeline, type PipelineConfig } from "../sim/pipeline"
import { buildTestJob } from "../sim/testJob"

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
export function runMemoryTilingSweep(
  pipeline: BurnSimPipeline,
  renderer: THREE.WebGLRenderer,
  sizesMm: number[] = DEFAULT_SIZES_MM,
): MemorySweepStep[] {
  const steps: MemorySweepStep[] = []
  const gl = renderer.getContext()

  for (const sizeMm of sizesMm) {
    const gridSize = Math.round(sizeMm / DIODE_PITCH_MM)
    const estimatedBytes = estimateResidentBytes(gridSize)

    if (gl.isContextLost()) {
      steps.push({ sizeMm, gridSize, success: false, contextLost: true, error: "context already lost; skipped", totalMs: null, estimatedBytes })
      continue
    }

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
      const result = pipeline.run(job, config)
      result.renderTarget.dispose()
      const lost = gl.isContextLost()
      steps.push({
        sizeMm,
        gridSize,
        success: !lost,
        contextLost: lost,
        error: lost ? "context lost during run" : null,
        totalMs: result.timings.totalMs,
        estimatedBytes,
      })
      if (lost) break
    } catch (err) {
      const lost = gl.isContextLost()
      steps.push({
        sizeMm,
        gridSize,
        success: false,
        contextLost: lost,
        error: err instanceof Error ? err.message : String(err),
        totalMs: null,
        estimatedBytes,
      })
      if (lost) break
    }
  }

  return steps
}
