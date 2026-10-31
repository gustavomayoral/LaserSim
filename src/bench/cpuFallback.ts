// CPU fallback timing (COD-252 §5): times the CPU/WASM deposit+blur path on
// the full-res test job, always — not only on GPUs missing EXT_float_blend —
// so the fallback's cost is a known number rather than an assumption.

import { runCpuReferenceTimed, type CpuReferenceResult } from "../sim/cpuReference"
import type { TestJob } from "../sim/testJob"

export interface CpuFallbackResult {
  gridSize: number
  depositMs: number
  blurHMs: number
  blurVMs: number
  totalMs: number
  // Per-stage sums from the same run, reused as the CPU column of the
  // full-res fluence-conservation table so the 10s CPU pass isn't run twice.
  stageFluence: CpuReferenceResult
}

export function runCpuFallbackBenchmark(
  job: TestJob,
  sizeMm: number,
  gridSize: number,
  spotSigmaMm: number,
  heatSigmaMm: number,
): CpuFallbackResult {
  const sigmaTotalMm = Math.sqrt(spotSigmaMm * spotSigmaMm + heatSigmaMm * heatSigmaMm)
  const timed = runCpuReferenceTimed(job, sizeMm, gridSize, sigmaTotalMm)
  return {
    gridSize,
    depositMs: timed.depositMs,
    blurHMs: timed.blurHMs,
    blurVMs: timed.blurVMs,
    totalMs: timed.totalMs,
    stageFluence: timed.result,
  }
}
