// COD-251 test job: a 100x100mm raster fill used to drive the deposit/blur/ramp
// pipeline. Velocity follows a trapezoidal profile at each line's ends (no full
// motion planner) so the slow-down near line ends deposits visibly more energy
// there, the way real end-of-line scorching looks.

export interface JobParams {
  sizeMm: number // square job size (mm)
  lineIntervalMm: number // raster line spacing (mm)
  gridPitchMm: number // sample spacing along each line == target grid pitch (mm)
  vMaxMmS: number // cruise velocity (mm/s)
  accelMmS2: number // accel/decel rate (mm/s^2)
  power: number // normalized laser power (arbitrary units)
  absorption: number // normalized material absorption (arbitrary units)
}

export const defaultJobParams: JobParams = {
  sizeMm: 100,
  lineIntervalMm: 0.1,
  gridPitchMm: 0.05 / 3,
  vMaxMmS: 500,
  accelMmS2: 20000,
  power: 1,
  absorption: 1,
}

// samples: interleaved [x, y, energy] in mm/mm/arbitrary-energy-units
export interface TestJob {
  samples: Float32Array
  sampleCount: number
  lineCount: number
  samplesPerLine: number
}

// Trapezoidal velocity at distance x along a line of length L: ramps up from a
// small floor velocity (never fully stops — the diode is still dwelling, which
// is exactly what produces end-of-line scorching) over accelDistance, cruises
// at vMax, then ramps back down symmetrically.
function velocityAt(x: number, length: number, vMax: number, accel: number): number {
  const vMin = vMax * 0.02
  const accelDistance = Math.min(length / 2, (vMax * vMax) / (2 * accel))
  let v: number
  if (x < accelDistance) v = Math.sqrt(2 * accel * x)
  else if (x > length - accelDistance) v = Math.sqrt(2 * accel * (length - x))
  else v = vMax
  return Math.max(v, vMin)
}

export function buildTestJob(params: JobParams = defaultJobParams): TestJob {
  const { sizeMm, lineIntervalMm, gridPitchMm, vMaxMmS, accelMmS2, power, absorption } = params
  const lineCount = Math.max(1, Math.round(sizeMm / lineIntervalMm))
  const samplesPerLine = Math.max(2, Math.round(sizeMm / gridPitchMm))
  const sampleCount = lineCount * samplesPerLine
  const samples = new Float32Array(sampleCount * 3)

  let i = 0
  for (let line = 0; line < lineCount; line++) {
    const y = (line + 0.5) * lineIntervalMm
    // Serpentine direction so consecutive lines don't all scorch on the same side.
    const leftToRight = line % 2 === 0
    for (let s = 0; s < samplesPerLine; s++) {
      const xAlong = (s / (samplesPerLine - 1)) * sizeMm
      const x = leftToRight ? xAlong : sizeMm - xAlong
      const v = velocityAt(xAlong, sizeMm, vMaxMmS, accelMmS2)
      const dt = gridPitchMm / v
      const energy = power * dt * absorption
      samples[i++] = x
      samples[i++] = y
      samples[i++] = energy
    }
  }

  return { samples, sampleCount, lineCount, samplesPerLine }
}
