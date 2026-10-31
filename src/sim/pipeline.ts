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
  maxRenderbufferSize: number
  colorBufferFloat: boolean
  floatBlend: boolean
  textureFloatLinear: boolean
  disjointTimerQuery: boolean
  renderer: string | null
  vendor: string | null
}

export function readGpuCapabilities(renderer: THREE.WebGLRenderer): GpuCapabilities {
  const gl = renderer.getContext()
  const debugInfo = gl.getExtension("WEBGL_debug_renderer_info")
  return {
    isWebGL2: renderer.capabilities.isWebGL2,
    maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
    maxRenderbufferSize: gl.getParameter(gl.MAX_RENDERBUFFER_SIZE),
    colorBufferFloat: renderer.extensions.has("EXT_color_buffer_float"),
    floatBlend: renderer.extensions.has("EXT_float_blend"),
    textureFloatLinear: renderer.extensions.has("OES_texture_float_linear"),
    disjointTimerQuery: renderer.extensions.has("EXT_disjoint_timer_query_webgl2"),
    renderer: debugInfo ? (gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) as string) : null,
    vendor: debugInfo ? (gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) as string) : null,
  }
}

const FULLSCREEN_GEOMETRY = new THREE.PlaneGeometry(2, 2)
// Local corners of the per-sample deposit quad (shared across all
// instances) — see the comment on depositVertex in shaders.ts.
const DEPOSIT_QUAD_LOCAL_POSITIONS = new Float32Array([0, 0, 1, 0, 0, 1, 1, 1])
const DEPOSIT_QUAD_INDICES = new Uint32Array([0, 1, 2, 2, 1, 3])

// RGBAFormat, not RedFormat: additive-blending into a single-channel float
// render target silently dropped most blended contributions on the GPU this
// was tested on (measured ~25% of the expected deposited energy — on this
// GPU EXT_float_blend evidently doesn't reliably cover the RED-only case,
// even though the extension itself reports available). This is exactly the
// platform-dependent fluence loss COD-251/COD-252 describe: it would vary
// by GPU/driver. RGBA costs more memory for an unused G/B/A channel but
// blends reliably; the shaders only ever read/write the R channel.
function makeRenderTarget(size: number): THREE.WebGLRenderTarget {
  const target = new THREE.WebGLRenderTarget(size, size, {
    format: THREE.RGBAFormat,
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
    const renderer = this.renderer
    // renderer.setViewport() always multiplies by the renderer's pixel ratio
    // (see WebGLRenderer.js), even when a render target is bound. The
    // on-screen heightmap scene sets pixel ratio to devicePixelRatio (e.g. 2
    // on a Retina display), so every setViewport(0,0,gridSize,gridSize)
    // call below was actually sizing the GL viewport to gridSize*ratio on
    // each axis — up to 4x the pixel count of the gridSize^2 render target
    // it's drawing into. That mismatch clips/misplaces most deposited
    // samples, and is platform/device dependent (different displays report
    // different ratios) — this is the real cause of the platform-dependent
    // fluence loss from COD-251, not just point-rasterization ambiguity.
    // Render the whole offscreen pipeline at ratio 1 and restore afterward.
    const previousPixelRatio = renderer.getPixelRatio()
    renderer.setPixelRatio(1)
    try {
      return this.runAtPixelRatioOne(job, config)
    } finally {
      renderer.setPixelRatio(previousPixelRatio)
    }
  }

  private runAtPixelRatioOne(job: TestJob, config: PipelineConfig): PipelineResult {
    const t0 = performance.now()
    const { sizeMm, gridSize, spotSigmaMm, heatSigmaMm, rampScale, ablationThreshold } = config
    const renderer = this.renderer

    // --- deposit pass ---
    // One quad instance per sample, spanning exactly the 2x2 texels that
    // sample's bilinear splat covers — see the comment on depositVertex in
    // shaders.ts for why this replaced GL_POINTS point-sprite rendering.
    const geometry = new THREE.InstancedBufferGeometry()
    geometry.setIndex(new THREE.BufferAttribute(DEPOSIT_QUAD_INDICES, 1))
    geometry.setAttribute("position", new THREE.BufferAttribute(DEPOSIT_QUAD_LOCAL_POSITIONS, 2))
    const centers = new Float32Array(job.sampleCount * 2)
    const energies = new Float32Array(job.sampleCount)
    for (let i = 0; i < job.sampleCount; i++) {
      centers[i * 2] = job.samples[i * 3]
      centers[i * 2 + 1] = job.samples[i * 3 + 1]
      energies[i] = job.samples[i * 3 + 2]
    }
    geometry.setAttribute("aCenter", new THREE.InstancedBufferAttribute(centers, 2))
    geometry.setAttribute("aEnergy", new THREE.InstancedBufferAttribute(energies, 1))
    geometry.instanceCount = job.sampleCount

    const depositMaterial = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: depositVertex,
      fragmentShader: depositFragment,
      uniforms: {
        uDomainMm: { value: new THREE.Vector2(sizeMm, sizeMm) },
        uGridSize: { value: new THREE.Vector2(gridSize, gridSize) },
      },
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false,
      blending: THREE.CustomBlending,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneFactor,
      blendEquation: THREE.AddEquation,
      transparent: true,
    })
    const depositMesh = new THREE.Mesh(geometry, depositMaterial)
    depositMesh.frustumCulled = false
    // position is a local 0/1 quad corner (not a real 3D vertex), so
    // three.js's computeBoundingSphere (which assumes itemSize 3) would read
    // garbage and produce NaN. It's still called once per render for z-sort
    // ordering even with frustumCulled off, so set a correct sphere directly.
    geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(sizeMm / 2, sizeMm / 2, 0), sizeMm)
    const depositScene = new THREE.Scene()
    depositScene.add(depositMesh)

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

    const totalFluence = sumFluenceExact(renderer, blurB, gridSize)

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

// Reads back the whole fluence target and sums it exactly, for the
// fluence-conservation check (COD-252 §3). This used to read back a fixed
// 20mm^2 corner window and scale it up to an estimated grid total, which is
// only valid for a spatially uniform field — this job's energy is
// concentrated near raster line ends, so a corner sample is nowhere near
// representative of the grid average and produced a meaningless "total".
// Reads back in row strips rather than one gridSize^2*4-float buffer, to
// bound peak memory at full-res (a 6000^2 RGBA float buffer would be ~576MB).
const FLUENCE_READBACK_ROWS_PER_CHUNK = 256
function sumFluenceExact(renderer: THREE.WebGLRenderer, target: THREE.WebGLRenderTarget, gridSize: number): number {
  const rowsPerChunk = Math.min(gridSize, FLUENCE_READBACK_ROWS_PER_CHUNK)
  const buffer = new Float32Array(gridSize * rowsPerChunk * 4)
  let sum = 0
  for (let y = 0; y < gridSize; y += rowsPerChunk) {
    const rows = Math.min(rowsPerChunk, gridSize - y)
    renderer.readRenderTargetPixels(target, 0, y, gridSize, rows, buffer)
    const texelCount = gridSize * rows
    for (let i = 0; i < texelCount; i++) sum += buffer[i * 4]
  }
  return sum
}
