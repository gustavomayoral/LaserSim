import * as THREE from "three"
import {
  depositVertex,
  depositFragment,
  fullscreenVertex,
  buildBlurFragment,
  rampFragment,
} from "./shaders"
import type { TestJob } from "./testJob"

export interface PipelineConfig {
  sizeMm: number
  gridSize: number // texels per side (clamped to GPU limits)
  spotSigmaMm: number
  heatSigmaMm: number // placeholder thermal-diffusion contribution; not calibrated
  rampScale: number // visualization "exposure" knob
  ablationThreshold: number // normalized 0..1, after rampScale
}

export interface PassTimings {
  depositMs: number
  blurMs: number
  rampMs: number
  totalMs: number
}

export interface PipelineResult {
  // The caller owns this target's lifetime (dispose() frees ~gridSize^2 * 4
  // bytes of GPU memory — significant at full-res) once it stops using it.
  renderTarget: THREE.WebGLRenderTarget
  timings: PassTimings
  gridSize: number
  totalFluence: number // sum over the grid, for the coarse/full-res consistency check
}

export interface GpuCapabilities {
  isWebGL2: boolean
  maxTextureSize: number
  colorBufferFloat: boolean
  floatBlend: boolean
}

export function readGpuCapabilities(renderer: THREE.WebGLRenderer): GpuCapabilities {
  const gl = renderer.getContext()
  return {
    isWebGL2: renderer.capabilities.isWebGL2,
    maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
    colorBufferFloat: renderer.extensions.has("EXT_color_buffer_float"),
    floatBlend: renderer.extensions.has("EXT_float_blend"),
  }
}

const FULLSCREEN_GEOMETRY = new THREE.PlaneGeometry(2, 2)

function makeRenderTarget(size: number): THREE.WebGLRenderTarget {
  const target = new THREE.WebGLRenderTarget(size, size, {
    format: THREE.RedFormat,
    type: THREE.FloatType,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    depthBuffer: false,
    stencilBuffer: false,
    wrapS: THREE.ClampToEdgeWrapping,
    wrapT: THREE.ClampToEdgeWrapping,
  })
  return target
}

// Runs deposit -> separable blur -> ramp for one job at one grid resolution.
// Render targets are allocated per call (a spike-level simplicity trade-off —
// a production version would pool/reuse them across runs of the same size).
export class BurnSimPipeline {
  private renderer: THREE.WebGLRenderer
  private orthoCamera: THREE.OrthographicCamera
  private fullscreenCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, -1, 1)

  constructor(renderer: THREE.WebGLRenderer) {
    this.renderer = renderer
    this.orthoCamera = new THREE.OrthographicCamera(0, 1, 1, 0, -1, 1)
  }

  run(job: TestJob, config: PipelineConfig): PipelineResult {
    const t0 = performance.now()
    const { sizeMm, gridSize, spotSigmaMm, heatSigmaMm, rampScale, ablationThreshold } = config
    const renderer = this.renderer

    // --- deposit pass ---
    const geometry = new THREE.BufferGeometry()
    const positions = new Float32Array(job.sampleCount * 2)
    const energies = new Float32Array(job.sampleCount)
    for (let i = 0; i < job.sampleCount; i++) {
      positions[i * 2] = job.samples[i * 3]
      positions[i * 2 + 1] = job.samples[i * 3 + 1]
      energies[i] = job.samples[i * 3 + 2]
    }
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 2))
    geometry.setAttribute("aEnergy", new THREE.BufferAttribute(energies, 1))

    const depositMaterial = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: depositVertex,
      fragmentShader: depositFragment,
      uniforms: {
        uDomainMm: { value: new THREE.Vector2(sizeMm, sizeMm) },
        uGridSize: { value: new THREE.Vector2(gridSize, gridSize) },
      },
      depthTest: false,
      depthWrite: false,
      blending: THREE.CustomBlending,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneFactor,
      blendEquation: THREE.AddEquation,
      transparent: true,
    })
    const points = new THREE.Points(geometry, depositMaterial)
    points.frustumCulled = false
    // position is vec2 (mm, not a real 3D vertex), so three.js's
    // computeBoundingSphere (which assumes itemSize 3) would read garbage and
    // produce NaN. It's still called once per render for z-sort ordering even
    // with frustumCulled off, so set a correct sphere directly instead.
    geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(sizeMm / 2, sizeMm / 2, 0), sizeMm)
    const depositScene = new THREE.Scene()
    depositScene.add(points)

    this.orthoCamera.left = 0
    this.orthoCamera.right = sizeMm
    this.orthoCamera.top = sizeMm
    this.orthoCamera.bottom = 0
    this.orthoCamera.updateProjectionMatrix()

    const depositTarget = makeRenderTarget(gridSize)
    renderer.setRenderTarget(depositTarget)
    renderer.setViewport(0, 0, gridSize, gridSize)
    renderer.setClearColor(0x000000, 0)
    renderer.clear()
    renderer.render(depositScene, this.orthoCamera)
    const tDeposit = performance.now()

    // --- separable Gaussian blur ---
    const sigmaTotalMm = Math.sqrt(spotSigmaMm * spotSigmaMm + heatSigmaMm * heatSigmaMm)
    const pitchMm = sizeMm / gridSize
    const sigmaTexels = Math.max(sigmaTotalMm / pitchMm, 1e-3)
    const radius = Math.min(64, Math.max(1, Math.ceil(3 * sigmaTexels)))
    let sumWeights = 0
    for (let k = -radius; k <= radius; k++) sumWeights += Math.exp(-0.5 * (k / sigmaTexels) ** 2)
    const invNorm = 1 / sumWeights

    const blurA = makeRenderTarget(gridSize)
    const blurB = makeRenderTarget(gridSize)
    const blurFragment = buildBlurFragment(radius)
    const blurMaterialH = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: fullscreenVertex,
      fragmentShader: blurFragment,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        uTex: { value: depositTarget.texture },
        uTexelStep: { value: new THREE.Vector2(1 / gridSize, 0) },
        uSigmaTexels: { value: sigmaTexels },
        uInvNorm: { value: invNorm },
      },
    })
    // Built fresh rather than via blurMaterialH.clone(): RawShaderMaterial's
    // cloneUniforms() can't clone texture-valued uniforms (it warns and drops
    // them), which left the second pass sampling an empty texture.
    const blurMaterialV = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: fullscreenVertex,
      fragmentShader: blurFragment,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        uTex: { value: blurA.texture },
        uTexelStep: { value: new THREE.Vector2(0, 1 / gridSize) },
        uSigmaTexels: { value: sigmaTexels },
        uInvNorm: { value: invNorm },
      },
    })

    const fullscreenMeshH = new THREE.Mesh(FULLSCREEN_GEOMETRY, blurMaterialH)
    const fullscreenSceneH = new THREE.Scene()
    fullscreenSceneH.add(fullscreenMeshH)
    renderer.setRenderTarget(blurA)
    renderer.setViewport(0, 0, gridSize, gridSize)
    renderer.render(fullscreenSceneH, this.fullscreenCamera)

    const fullscreenMeshV = new THREE.Mesh(FULLSCREEN_GEOMETRY, blurMaterialV)
    const fullscreenSceneV = new THREE.Scene()
    fullscreenSceneV.add(fullscreenMeshV)
    renderer.setRenderTarget(blurB)
    renderer.setViewport(0, 0, gridSize, gridSize)
    renderer.render(fullscreenSceneV, this.fullscreenCamera)
    const tBlur = performance.now()

    // --- material ramp ---
    const rampTarget = new THREE.WebGLRenderTarget(gridSize, gridSize, {
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
      stencilBuffer: false,
      wrapS: THREE.ClampToEdgeWrapping,
      wrapT: THREE.ClampToEdgeWrapping,
    })
    const rampMaterial = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: fullscreenVertex,
      fragmentShader: rampFragment,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        uFluence: { value: blurB.texture },
        uScale: { value: rampScale },
        uThreshold: { value: ablationThreshold },
      },
    })
    const rampMesh = new THREE.Mesh(FULLSCREEN_GEOMETRY, rampMaterial)
    const rampScene = new THREE.Scene()
    rampScene.add(rampMesh)
    renderer.setRenderTarget(rampTarget)
    renderer.setViewport(0, 0, gridSize, gridSize)
    renderer.render(rampScene, this.fullscreenCamera)
    renderer.setRenderTarget(null)
    const tRamp = performance.now()

    const totalFluence = sumFluence(renderer, blurB, gridSize, pitchMm)

    // Dispose everything except the ramp target the caller needs to sample.
    geometry.dispose()
    depositMaterial.dispose()
    depositTarget.dispose()
    blurMaterialH.dispose()
    blurMaterialV.dispose()
    blurA.dispose()
    blurB.dispose()
    rampMaterial.dispose()

    return {
      renderTarget: rampTarget,
      gridSize,
      totalFluence,
      timings: {
        depositMs: tDeposit - t0,
        blurMs: tBlur - tDeposit,
        rampMs: tRamp - tBlur,
        totalMs: tRamp - t0,
      },
    }
  }
}

// Reads back a fixed *physical* corner region (not a fixed pixel count) of
// the blurred fluence target and scales it up to an estimated grid total —
// used only for the coarse/full-res "visually consistent" sanity check, not
// the hot path. Sampling the same mm^2 window at both resolutions is what
// makes the two numbers comparable despite very different pixel counts.
const FLUENCE_SAMPLE_WINDOW_MM = 20
function sumFluence(
  renderer: THREE.WebGLRenderer,
  target: THREE.WebGLRenderTarget,
  gridSize: number,
  pitchMm: number,
): number {
  const sampleDim = Math.min(gridSize, Math.max(1, Math.round(FLUENCE_SAMPLE_WINDOW_MM / pitchMm)))
  const buffer = new Float32Array(sampleDim * sampleDim * 4)
  renderer.readRenderTargetPixels(target, 0, 0, sampleDim, sampleDim, buffer)
  let sum = 0
  for (let i = 0; i < sampleDim * sampleDim; i++) sum += buffer[i * 4]
  const scale = (gridSize * gridSize) / (sampleDim * sampleDim)
  return sum * scale
}
