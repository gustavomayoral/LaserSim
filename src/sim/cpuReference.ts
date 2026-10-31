// CPU reference implementation of deposit + separable Gaussian blur.
//
// This mirrors the GPU math in shaders.ts/pipeline.ts exactly (same corner
// weights, same kernel), so it serves three purposes at once (COD-252 §3):
//   - test oracle: GPU output should match this within floating-point error
//   - fallback path when EXT_float_blend is missing (no GPU additive blend)
//   - headless path for Vitest, where no WebGL context exists
//
// Not optimized — O(sampleCount) deposit and O(gridSize^2 * radius) blur,
// fine for the fluence-conservation checks this is used for.

import type { TestJob } from "./testJob"

export function cpuDeposit(job: TestJob, sizeMm: number, gridSize: number): Float32Array {
  const grid = new Float32Array(gridSize * gridSize)
  const { samples, sampleCount } = job
  for (let s = 0; s < sampleCount; s++) {
    const xMm = samples[s * 3]
    const yMm = samples[s * 3 + 1]
    const energy = samples[s * 3 + 2]

    const px = (xMm / sizeMm) * gridSize
    const py = (yMm / sizeMm) * gridSize
    const i0x = Math.floor(px - 0.5)
    const i0y = Math.floor(py - 0.5)
    const fracX = px - (i0x + 0.5)
    const fracY = py - (i0y + 0.5)

    // Clamp (not discard) out-of-bounds corners — see the matching comment
    // in shaders.ts's depositVertex for why this must conserve, not drop,
    // energy from samples near the domain edge.
    for (const dy of [0, 1]) {
      const ty = clamp(i0y + dy, 0, gridSize - 1)
      const wy = dy === 0 ? 1 - fracY : fracY
      for (const dx of [0, 1]) {
        const tx = clamp(i0x + dx, 0, gridSize - 1)
        const wx = dx === 0 ? 1 - fracX : fracX
        grid[ty * gridSize + tx] += energy * wx * wy
      }
    }
  }
  return grid
}

// Separable Gaussian blur pass, one axis at a time — same kernel/truncation
// radius and normalization as buildBlurFragment in shaders.ts. Edge texels
// use clamp-to-edge sampling, matching the GPU render target's wrap mode.
export function cpuBlurPass(grid: Float32Array, gridSize: number, sigmaTexels: number, axis: "x" | "y"): Float32Array {
  const radius = Math.min(64, Math.max(1, Math.ceil(3 * sigmaTexels)))
  const weights = new Float32Array(2 * radius + 1)
  let sumWeights = 0
  for (let k = -radius; k <= radius; k++) {
    const w = Math.exp(-0.5 * (k / sigmaTexels) ** 2)
    weights[k + radius] = w
    sumWeights += w
  }
  const invNorm = 1 / sumWeights

  // Zero-pad out-of-domain taps rather than clamping to the edge texel — see
  // the matching comment on the GPU blur fragment shader in shaders.ts for
  // why clamping over-counts energy concentrated near the domain boundary.
  const out = new Float32Array(gridSize * gridSize)
  for (let y = 0; y < gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      let sum = 0
      for (let k = -radius; k <= radius; k++) {
        const sx = axis === "x" ? x + k : x
        const sy = axis === "y" ? y + k : y
        if (sx < 0 || sx >= gridSize || sy < 0 || sy >= gridSize) continue
        sum += grid[sy * gridSize + sx] * weights[k + radius]
      }
      out[y * gridSize + x] = sum * invNorm
    }
  }
  return out
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v
}

export function sumGrid(grid: Float32Array): number {
  let sum = 0
  for (let i = 0; i < grid.length; i++) sum += grid[i]
  return sum
}

export interface CpuReferenceResult {
  afterDeposit: number
  afterBlurH: number
  afterBlurV: number
}

// Runs the full deposit -> blurH -> blurV chain on the CPU and reports the
// grid-wide energy sum after each stage, for the fluence-conservation table.
export function runCpuReference(
  job: TestJob,
  sizeMm: number,
  gridSize: number,
  sigmaTotalMm: number,
): CpuReferenceResult {
  return runCpuReferenceTimed(job, sizeMm, gridSize, sigmaTotalMm).result
}

export interface CpuReferenceTimedResult {
  result: CpuReferenceResult
  depositMs: number
  blurHMs: number
  blurVMs: number
  totalMs: number
}

// Same as runCpuReference, but also times each stage with performance.now()
// — used for the CPU-fallback cost measurement (COD-252 §5): this is the
// path EXT_float_blend-less GPUs (or a from-scratch headless run) would pay,
// so its cost needs to be known even on machines where the GPU path works.
export function runCpuReferenceTimed(
  job: TestJob,
  sizeMm: number,
  gridSize: number,
  sigmaTotalMm: number,
): CpuReferenceTimedResult {
  const pitchMm = sizeMm / gridSize
  const sigmaTexels = Math.max(sigmaTotalMm / pitchMm, 1e-3)

  const t0 = performance.now()
  const deposited = cpuDeposit(job, sizeMm, gridSize)
  const t1 = performance.now()
  const afterDeposit = sumGrid(deposited)

  const blurredH = cpuBlurPass(deposited, gridSize, sigmaTexels, "x")
  const t2 = performance.now()
  const afterBlurH = sumGrid(blurredH)

  const blurredV = cpuBlurPass(blurredH, gridSize, sigmaTexels, "y")
  const t3 = performance.now()
  const afterBlurV = sumGrid(blurredV)

  return {
    result: { afterDeposit, afterBlurH, afterBlurV },
    depositMs: t1 - t0,
    blurHMs: t2 - t1,
    blurVMs: t3 - t2,
    totalMs: t3 - t0,
  }
}
