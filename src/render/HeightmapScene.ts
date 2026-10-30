import * as THREE from "three"
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js"
import { heightmapVertex, heightmapFragment } from "../sim/shaders"

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

    const loop = () => {
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
