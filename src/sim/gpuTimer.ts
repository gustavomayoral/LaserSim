// GPU timing correctness helpers (COD-252 §2).
//
// Issuing GL draw calls only enqueues work — performance.now() around
// renderer.render() alone measures submission latency, not execution time
// (COD-251's numbers were "measure the queue is non-empty", not real
// timings). Every timed stage must sync before its wall-clock end is taken,
// and should also attempt a GPU-side measurement via
// EXT_disjoint_timer_query_webgl2 when available, since wall-clock after a
// sync includes CPU-side readback/driver overhead on top of actual GPU time.

export interface StageTime {
  wallMs: number
  // null when EXT_disjoint_timer_query_webgl2 is unavailable, the query
  // result never became available, or the driver reported a disjoint event
  // (timing invalidated by a GPU state change mid-query).
  gpuMs: number | null
}

type TimerExt = {
  TIME_ELAPSED_EXT: number
  GPU_DISJOINT_EXT: number
  beginQueryEXT: (target: number, query: WebGLQuery) => void
  endQueryEXT: (target: number) => void
}

export function getTimerExtension(gl: WebGL2RenderingContext): TimerExt | null {
  return gl.getExtension("EXT_disjoint_timer_query_webgl2") as TimerExt | null
}

// Blocks until all GL commands issued so far have completed on the GPU, then
// does a 1x1 readback as belt-and-suspenders: on some driver/backend
// combinations (notably ANGLE's D3D11 path on Windows) gl.finish() has been
// observed to return before the swapchain/readback path is actually
// coherent, so a tiny readPixels forces a real synchronization point. The
// 1x1 read is from whichever framebuffer is currently bound (the stage's own
// render target), so it doesn't disturb the pipeline's output.
export function gpuSync(gl: WebGL2RenderingContext): void {
  gl.finish()
  const px = new Uint8Array(4)
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px)
}

const QUERY_POLL_TIMEOUT_MS = 50

// Times one GPU stage: begins a timer query (if the extension is available),
// runs `render` (which issues the stage's draw call(s)), syncs the GPU, ends
// the query, and polls for its result. Call `gpuSync` is done here, and only
// here, so stage boundaries are real completion points, not just submission
// points.
export function timeStage(gl: WebGL2RenderingContext, ext: TimerExt | null, render: () => void): StageTime {
  const query = ext ? gl.createQuery() : null
  if (ext && query) ext.beginQueryEXT(ext.TIME_ELAPSED_EXT, query)

  const t0 = performance.now()
  render()
  gpuSync(gl)
  const wallMs = performance.now() - t0

  if (ext && query) ext.endQueryEXT(ext.TIME_ELAPSED_EXT)

  let gpuMs: number | null = null
  if (ext && query) {
    const pollStart = performance.now()
    while (performance.now() - pollStart < QUERY_POLL_TIMEOUT_MS) {
      if (gl.getParameter(ext.GPU_DISJOINT_EXT)) {
        gpuMs = null
        break
      }
      if (gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE)) {
        const ns = gl.getQueryParameter(query, gl.QUERY_RESULT) as number
        gpuMs = ns / 1e6
        break
      }
    }
    gl.deleteQuery(query)
  }

  return { wallMs, gpuMs }
}

export function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
}

export function minMaxMedian(values: number[]): { min: number; max: number; median: number } {
  return { min: Math.min(...values), max: Math.max(...values), median: median(values) }
}
