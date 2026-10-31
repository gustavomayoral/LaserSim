import * as THREE from "three"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"
import { heightmapVertex, heightmapFragment } from "../sim/shaders"

// Result of measureFps(). Frame times are rAF-to-rAF intervals, so they're
// capped below by the display's refresh interval (16.7ms at 60Hz) — a
// GPU-bound frame shows up as a longer interval, not a higher fps.
export interface FpsResult {
  durationMs: number
  frames: number
  meanFps: number
  frameMsP50: number
  frameMsP95: number
  frameMsMax: number
  // Intervals over 1.5x the 60Hz budget (25ms), i.e. visibly dropped frames.
  longFrames: number
  textureGridSize: number | null
  // Drawing-buffer size in device pixels: fps scales with it, so a result
  // from the default 800x600 window isn't comparable to a maximized one.
  canvasPx: { width: number; height: number }
  // rAF is throttled or paused while the page is hidden or the window is
  // occluded/minimized, which makes the numbers meaningless; flagged rather
  // than silently reported.
  pageHiddenDuringRun: boolean
}

interface FpsProbe {
  startMs: number
  durationMs: number
  lastMs: number | null
  intervals: number[]
  hidden: boolean
  radius: number
  height: number
  startAngle: number
  resolve: (result: FpsResult) => void
}

// Owns the three.js scene/camera/controls for the heightmap preview. Kept as
// a plain class (not a React hook) so App.tsx can create one on mount and
// just hand it fresh pipeline output each run.
export class HeightmapScene {
  readonly renderer: THREE.WebGLRenderer
  private scene = new THREE.Scene()
  private camera: THREE.PerspectiveCamera
  private controls: OrbitControls
  private mesh: THREE.Mesh
  private material: THREE.RawShaderMaterial
  private animHandle = 0
  private ownedTarget: THREE.WebGLRenderTarget | null = null
  private placeholderTex: THREE.DataTexture
  private displaySize = new THREE.Vector2(1, 1)
  private fpsProbe: FpsProbe | null = null
  private textureGridSize: number | null = null

  constructor(canvas: HTMLCanvasElement, sizeMm: number) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.displaySize.set(canvas.clientWidth || 1, canvas.clientHeight || 1)
    this.scene.background = new THREE.Color(0x1b1d22)

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 2000)
    this.camera.position.set(sizeMm * 0.8, sizeMm * 0.9, sizeMm * 0.8)

    this.controls = new OrbitControls(this.camera, canvas)
    this.controls.target.set(0, 0, 0)
    this.controls.enableDamping = true
    this.controls.minDistance = sizeMm * 0.2
    this.controls.maxDistance = sizeMm * 4

    this.placeholderTex = new THREE.DataTexture(new Uint8Array([200, 160, 110, 0]), 1, 1)
    this.placeholderTex.needsUpdate = true

    this.material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: heightmapVertex,
      fragmentShader: heightmapFragment,
      uniforms: {
        uColorHeight: { value: this.placeholderTex },
        uHeightScale: { value: 6 },
        uTexelStep: { value: new THREE.Vector2(1 / 512, 1 / 512) },
        uDomainMm: { value: new THREE.Vector2(sizeMm, sizeMm) },
        uLightDirWorld: { value: new THREE.Vector3(0.4, 0.8, 0.5) },
      },
      side: THREE.DoubleSide,
    })

    // PlaneGeometry is centered at its own origin with uv (0,0)-(1,1) across
    // it regardless of world position, so centering it under the camera
    // target doesn't affect which texel each uv samples.
    const geometry = new THREE.PlaneGeometry(sizeMm, sizeMm, 400, 400)
    this.mesh = new THREE.Mesh(geometry, this.material)
    this.mesh.rotation.x = -Math.PI / 2
    this.scene.add(this.mesh)

    const loop = (now: number) => {
      this.stepFpsProbe(now)
      this.controls.update()
      try {
        // BurnSimPipeline renders its offscreen passes on this same shared
        // renderer and leaves the viewport set to the simulation grid size
        // (up to 6000x6000) — reset it to the canvas size every frame rather
        // than trying to track every place the pipeline could run from.
        this.renderer.setViewport(0, 0, this.displaySize.x, this.displaySize.y)
        this.renderer.render(this.scene, this.camera)
      } catch (err) {
        // Keep the rAF chain alive even if one frame throws (e.g. a texture
        // swapped mid-frame) instead of silently freezing the preview.
        console.error("HeightmapScene render error", err)
      }
      this.animHandle = requestAnimationFrame(loop)
    }
    this.animHandle = requestAnimationFrame(loop)
  }

  // Takes ownership of `target` (disposing it, GPU memory and all, the next
  // time setTexture or dispose() runs) — callers must not reuse or dispose it
  // themselves. Full-res targets are ~150MB each; leaving old ones undisposed
  // exhausted GPU memory within a handful of runs during testing.
  setTexture(target: THREE.WebGLRenderTarget, gridSize: number) {
    this.material.uniforms.uColorHeight.value = target.texture
    this.material.uniforms.uTexelStep.value = new THREE.Vector2(1 / gridSize, 1 / gridSize)
    this.ownedTarget?.dispose()
    this.ownedTarget = target
    this.textureGridSize = gridSize
  }

  // Heightmap fps check (COD-252 pass criterion "heightmap ~60 fps"):
  // orbits the camera one full turn around the target over `durationMs`,
  // driven from the render loop itself so every frame both moves and
  // renders, and records each rAF-to-rAF interval. The camera is put back
  // where it was afterwards. User drags during the run are overridden.
  measureFps(durationMs = 5000): Promise<FpsResult> {
    if (this.fpsProbe) return Promise.reject(new Error("fps measurement already running"))
    const offset = this.camera.position.clone().sub(this.controls.target)
    const savedPosition = this.camera.position.clone()
    return new Promise<FpsResult>((resolve) => {
      this.fpsProbe = {
        startMs: performance.now(),
        durationMs,
        lastMs: null,
        intervals: [],
        hidden: document.hidden,
        radius: Math.hypot(offset.x, offset.z),
        height: offset.y,
        startAngle: Math.atan2(offset.z, offset.x),
        resolve: (result) => {
          this.camera.position.copy(savedPosition)
          resolve(result)
        },
      }
    })
  }

  private stepFpsProbe(now: number) {
    const probe = this.fpsProbe
    if (!probe) return
    probe.hidden ||= document.hidden
    if (probe.lastMs !== null) probe.intervals.push(now - probe.lastMs)
    probe.lastMs = now

    const elapsed = now - probe.startMs
    const angle = probe.startAngle + (2 * Math.PI * Math.min(elapsed, probe.durationMs)) / probe.durationMs
    const target = this.controls.target
    this.camera.position.set(
      target.x + probe.radius * Math.cos(angle),
      target.y + probe.height,
      target.z + probe.radius * Math.sin(angle),
    )

    if (elapsed < probe.durationMs) return
    this.fpsProbe = null
    const sorted = [...probe.intervals].sort((a, b) => a - b)
    const pick = (p: number) => (sorted.length > 0 ? sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))] : NaN)
    const total = probe.intervals.reduce((sum, ms) => sum + ms, 0)
    probe.resolve({
      durationMs: elapsed,
      frames: probe.intervals.length,
      meanFps: total > 0 ? (probe.intervals.length / total) * 1000 : 0,
      frameMsP50: pick(0.5),
      frameMsP95: pick(0.95),
      frameMsMax: sorted.length > 0 ? sorted[sorted.length - 1] : NaN,
      longFrames: probe.intervals.filter((ms) => ms > 25).length,
      textureGridSize: this.textureGridSize,
      canvasPx: { width: this.renderer.domElement.width, height: this.renderer.domElement.height },
      pageHiddenDuringRun: probe.hidden,
    })
  }

  resize(width: number, height: number) {
    this.camera.aspect = width / Math.max(height, 1)
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(width, height, false)
    // setViewport takes logical (CSS) pixels, same as setSize — it applies
    // the device pixel ratio itself, so don't multiply by it here.
    this.displaySize.set(width, height)
  }

  dispose() {
    cancelAnimationFrame(this.animHandle)
    this.controls.dispose()
    this.mesh.geometry.dispose()
    this.material.dispose()
    this.placeholderTex.dispose()
    this.ownedTarget?.dispose()
    this.renderer.dispose()
  }
}
