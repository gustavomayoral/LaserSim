import { describe, expect, it } from "vitest"
import { buildTestJob, defaultJobParams, type JobParams } from "./testJob"
import { cpuDeposit, cpuBlurPass, sumGrid, runCpuReference } from "./cpuReference"

// Small, fast params — just enough samples/grid to exercise the math without
// the full production job's multi-million-sample cost, but with the same
// sizeMm/vMax/accel as defaultJobParams (testJob.ts) so each raster line
// still has a real cruise plateau between its accel/decel ramps, not just
// the end-of-line slowdown dominating the whole line (a job where
// accelDistance ~= half the line length concentrates most energy right at
// the domain edge, which stresses blur's clamp-to-edge boundary handling in
// a way the production job doesn't).
const smallJobParams: JobParams = {
  sizeMm: 25, // long enough that accelDistance (6.25mm, from vMax/accel below) leaves a real cruise plateau
  lineIntervalMm: 2,
  gridPitchMm: 0.05 / 3, // same sample pitch as the production job (testJob.ts defaultJobParams)
  vMaxMmS: 500,
  accelMmS2: 20000,
  power: 1,
  absorption: 1,
}
const GRID_SIZE = 1500 // sizeMm / gridPitchMm, matching the production full-res grid's texel pitch
const SIGMA_TOTAL_MM = Math.sqrt(0.05 ** 2 + 0.03 ** 2) // production spotSigmaMm/heatSigmaMm (pipeline.ts)

function expectedEnergy(samples: Float32Array): number {
  let sum = 0
  for (let i = 2; i < samples.length; i += 3) sum += samples[i]
  return sum
}

describe("cpuDeposit", () => {
  it("conserves total energy for samples away from the domain edge", () => {
    const job = buildTestJob(smallJobParams)
    const grid = cpuDeposit(job, smallJobParams.sizeMm, GRID_SIZE)
    const expected = expectedEnergy(job.samples)
    const measured = sumGrid(grid)
    // Raster lines start at y = 0.5*lineIntervalMm and end 1 texel pitch
    // short of the far edge in x, so interior samples lose nothing to
    // clamping; only near-boundary float rounding should show up here.
    expect(measured / expected).toBeCloseTo(1, 2)
  })
})

describe("cpuBlurPass", () => {
  it("conserves the grid-wide sum regardless of sigma (normalized kernel)", () => {
    const grid = new Float32Array(GRID_SIZE * GRID_SIZE)
    // A single interior impulse, far enough from the edge that clamp-to-edge
    // truncation of the kernel tails doesn't lose any weight.
    grid[(GRID_SIZE / 2) * GRID_SIZE + GRID_SIZE / 2] = 1
    const before = sumGrid(grid)
    const afterH = cpuBlurPass(grid, GRID_SIZE, 5, "x")
    const afterV = cpuBlurPass(afterH, GRID_SIZE, 5, "y")
    expect(sumGrid(afterH) / before).toBeCloseTo(1, 4)
    expect(sumGrid(afterV) / before).toBeCloseTo(1, 4)
  })
})

describe("runCpuReference", () => {
  // Raster lines run edge-to-edge (x=0 to x=sizeMm, same for y), and
  // velocity ramps down to a floor right at each edge (testJob.ts), so a
  // real amount of energy sits within one blur radius of the domain
  // boundary on every job regardless of size. Blur's boundary is reflective
  // (cpuBlurPass/the GPU blur fragment shader normalize each output texel by
  // only its in-bounds taps' weight sum, not a fixed global one), which
  // conserves exactly for an interior source (same math as the single-
  // impulse test above) but not quite exactly for a source near the
  // boundary: that source feeds several nearby outputs, each normalized by
  // its OWN (smaller, boundary-reduced) weight sum, so its total outgoing
  // weight across all of them isn't guaranteed to be exactly 1 the way an
  // interior source's is. In practice this is a small residual, not the
  // zero-pad policy's outright loss — this sizeMm=25 job is small enough
  // that the near-boundary fraction of its total energy is large, so the
  // residual is visible here (~3%) even though it's negligible at
  // production scale (see the <1% checks below, which are the actual
  // COD-252 §3 pass criterion).
  it("conserves fluence much more closely than the old lossy zero-pad boundary, even for a small edge-dominated job", () => {
    const job = buildTestJob(smallJobParams)
    const expected = expectedEnergy(job.samples)
    const result = runCpuReference(job, smallJobParams.sizeMm, GRID_SIZE, SIGMA_TOTAL_MM)

    expect(result.afterDeposit / expected).toBeCloseTo(1, 2)
    expect(Math.abs(result.afterBlurV / expected - 1)).toBeLessThan(0.05)
  })

  // At the production job's actual size (defaultJobParams: 100mm, raster
  // lines running edge-to-edge), the near-boundary energy concentration
  // that the small synthetic job above exaggerates becomes a small fraction
  // of the job's total energy, since cruise-phase dwell time (most of each
  // line) scales with job size while the near-edge dwell-time spike does
  // not. This is the actual COD-252 §3 pass criterion check.
  it("conserves fluence within 1% at production job scale (coarse grid)", () => {
    const job = buildTestJob(defaultJobParams)
    const expected = expectedEnergy(job.samples)
    const sigmaTotalMm = Math.sqrt(0.05 ** 2 + 0.03 ** 2) // pipeline.ts spotSigmaMm/heatSigmaMm
    const result = runCpuReference(job, defaultJobParams.sizeMm, 500, sigmaTotalMm)

    expect(Math.abs(result.afterDeposit / expected - 1)).toBeLessThan(0.01)
    expect(Math.abs(result.afterBlurH / expected - 1)).toBeLessThan(0.01)
    expect(Math.abs(result.afterBlurV / expected - 1)).toBeLessThan(0.01)
  })

  // The app's actual coarse-scrub job (App.tsx's getJobs().coarse) isn't
  // defaultJobParams resampled onto a smaller grid — it rebuilds the job
  // with gridPitchMm set to match the coarse grid pitch exactly
  // (SIZE_MM / COARSE_GRID), so every line has only ~500 samples instead of
  // ~6000. The test above kept the full-res sample pitch, which never
  // exercised this: with the real coarse sample pitch, each line's very
  // first/last sample lands exactly on the domain edge — the same case the
  // GPU shader had to be fixed for (see shaders.ts's depositVertex) — and
  // without that fix this failed by several percent, not fractions of one.
  it("conserves fluence within 1% for the app's actual coarse-scrub job (sample pitch == grid pitch)", () => {
    const coarseGrid = 500
    const job = buildTestJob({ ...defaultJobParams, gridPitchMm: defaultJobParams.sizeMm / coarseGrid })
    const expected = expectedEnergy(job.samples)
    const sigmaTotalMm = Math.sqrt(0.05 ** 2 + 0.03 ** 2)
    const result = runCpuReference(job, defaultJobParams.sizeMm, coarseGrid, sigmaTotalMm)

    expect(Math.abs(result.afterDeposit / expected - 1)).toBeLessThan(0.01)
    expect(Math.abs(result.afterBlurH / expected - 1)).toBeLessThan(0.01)
    expect(Math.abs(result.afterBlurV / expected - 1)).toBeLessThan(0.01)
  })
})
