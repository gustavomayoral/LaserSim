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
  // (timing invalidated by a GPU state change mid-query). Also null until
  // resolveGpuTimes() has run — see timeStage.
  gpuMs: number | null
  // Timer query still waiting for its result; consumed (and deleted) by
  // resolveGpuTimes().
  pendingQuery?: WebGLQuery
}

// EXT_disjoint_timer_query_webgl2 only adds the TIME_ELAPSED_EXT/
// GPU_DISJOINT_EXT constants — WebGL2 already has native query objects, so
// the begin/end calls are the core gl.beginQuery/gl.endQuery, not an EXT-
// suffixed method on the extension object (that's the older WebGL1
// EXT_disjoint_timer_query's API shape, which this is not).
type TimerExt = {
  TIME_ELAPSED_EXT: number
  GPU_DISJOINT_EXT: number
}

export function getTimerExtension(gl: WebGL2RenderingContext): TimerExt | null {
  return gl.getExtension("EXT_disjoint_timer_query_webgl2") as TimerExt | null
}

// readPixels only guarantees RGBA/UNSIGNED_BYTE for a normalized 8-bit
// attachment; this pipeline's deposit/blur targets are RGBA32F, where that
// combination is GL_INVALID_OPERATION on ANGLE (confirmed: every sync was
// logging "glReadPixelsRobustANGLE: Invalid format and type combination").
// IMPLEMENTATION_COLOR_READ_FORMAT/TYPE report the pair that's actually
// valid for whatever is currently bound, so ask for that instead of
// assuming one.
function readbackBuffer(gl: WebGL2RenderingContext, type: number): ArrayBufferView {
  switch (type) {
    case gl.BYTE:
      return new Int8Array(4)
    case gl.UNSIGNED_SHORT:
      return new Uint16Array(4)
    case gl.SHORT:
      return new Int16Array(4)
    case gl.UNSIGNED_INT:
      return new Uint32Array(4)
    case gl.INT:
      return new Int32Array(4)
    case gl.FLOAT:
      return new Float32Array(4)
    case gl.UNSIGNED_BYTE:
    default:
      return new Uint8Array(4)
  }
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
  const format = gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_FORMAT) as number
  const type = gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_TYPE) as number
  try {
    gl.readPixels(0, 0, 1, 1, format, type, readbackBuffer(gl, type))
  } catch {
    // gl.finish() above already forced the real sync point; the readback
    // itself is only belt-and-suspenders, so a failure here (e.g. an
    // attachment combination IMPLEMENTATION_COLOR_READ_* didn't predict)
    // isn't fatal to timing correctness.
  }
}

// Times one GPU stage: begins a timer query (if the extension is available),
// runs `render` (which issues the stage's draw call(s)), ends the query, then
// syncs the GPU and takes the wall-clock end. The query's begin/end window
// covers only the actual draw call, not our own gl.finish()/readPixels sync
// calls — some drivers flag a timer query's result "disjoint" if a
// synchronizing call happens while it's still open, so ending it first keeps
// that risk out of the measurement it's trying to take.
//
// `sync = false` skips gl.finish()/readPixels and the timer query entirely,
// measuring only submission time (COD-251's original, naive method) — this
// is NOT a GPU-complete measurement, but it IS what real interactive use
// (e.g. a future live scrub control) would actually pay, since nothing
// forces the pipeline to synchronously block there. Call sites must label
// which one they're reporting; see report.ts.
//
// The timer query's result is NOT read here. WebGL only makes query results
// available after control returns to the browser's event loop (WebGL 2
// spec §5.32), so this used to busy-wait its full 50ms poll timeout on
// every stage and still get nothing: every report showed gpu=unavailable,
// and the 4 x 50ms of spinning was ~200ms of every pass's totalMs that no
// stage timer covered (the "missing" coarse-scrub time in the first
// Windows report). The query now rides along on the returned StageTime;
// call resolveGpuTimes() from an async context once the batch of runs is
// done.
export function timeStage(
  gl: WebGL2RenderingContext,
  ext: TimerExt | null,
  render: () => void,
  sync = true,
): StageTime {
  if (!sync) {
    const t0 = performance.now()
    render()
    return { wallMs: performance.now() - t0, gpuMs: null }
  }

  const query = ext ? gl.createQuery() : null
  if (ext && query) gl.beginQuery(ext.TIME_ELAPSED_EXT, query)

  const t0 = performance.now()
  render()
  if (ext && query) gl.endQuery(ext.TIME_ELAPSED_EXT)
  gpuSync(gl)
  const wallMs = performance.now() - t0

  return query ? { wallMs, gpuMs: null, pendingQuery: query } : { wallMs, gpuMs: null }
}

const QUERY_RESOLVE_TIMEOUT_MS = 2000

// Fills in gpuMs for every StageTime whose timer query is still pending,
// yielding to the event loop between polls (results can't become available
// otherwise). A disjoint event invalidates every query still outstanding,
// so those stay null. Always deletes the queries it was given.
export async function resolveGpuTimes(gl: WebGL2RenderingContext, stages: StageTime[]): Promise<void> {
  const ext = getTimerExtension(gl)
  let pending = stages.filter((s) => s.pendingQuery)
  const start = performance.now()
  while (pending.length > 0 && ext && performance.now() - start < QUERY_RESOLVE_TIMEOUT_MS) {
    await new Promise((r) => setTimeout(r, 0))
    if (gl.getParameter(ext.GPU_DISJOINT_EXT)) break
    pending = pending.filter((s) => {
      const query = s.pendingQuery!
      if (!gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE)) return true
      s.gpuMs = (gl.getQueryParameter(query, gl.QUERY_RESULT) as number) / 1e6
      gl.deleteQuery(query)
      delete s.pendingQuery
      return false
    })
  }
  for (const s of pending) {
    gl.deleteQuery(s.pendingQuery!)
    delete s.pendingQuery
  }
}

// For runs whose GPU times nobody will read (warmup, the memory sweep):
// frees their queries without waiting.
export function discardGpuTimes(gl: WebGL2RenderingContext, stages: StageTime[]): void {
  for (const s of stages) {
    if (s.pendingQuery) {
      gl.deleteQuery(s.pendingQuery)
      delete s.pendingQuery
    }
  }
}

export function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
}

export function minMaxMedian(values: number[]): { min: number; max: number; median: number } {
  return { min: Math.min(...values), max: Math.max(...values), median: median(values) }
}
