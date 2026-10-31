// IPC throughput/latency benchmark (COD-252 §6): drives the three listener
// modes exposed by src-tauri (per-line global events — the COD-251 baseline
// still running on gcode_listener's port; batched global events; and a
// Tauri v2 Channel) against the same `node scripts/send-gcode.mjs` sender,
// and reports throughput plus p50/p95 per-batch latency for each. Also
// supports measuring invoke() round-trip latency while a stream is in
// flight, for the "`?` status query under load" check.

export type IpcMode = "per-line-events" | "batched-events" | "channel"

export interface IpcModeResult {
  mode: IpcMode
  batchSize: number
  port: number
  lineCount: number
  durationMs: number
  linesPerSec: number
  latencyP50Ms: number
  latencyP95Ms: number
  statusRttWhileStreamingMs: { p50: number; p95: number } | null
}

export const IPC_PORTS: Record<IpcMode, number> = {
  "per-line-events": 8090, // gcode_listener.rs's always-on baseline listener
  "batched-events": 8091,
  channel: 8092,
}

function percentile(sortedAsc: number[], p: number): number {
  if (sortedAsc.length === 0) return NaN
  const idx = Math.min(sortedAsc.length - 1, Math.floor(p * sortedAsc.length))
  return sortedAsc[idx]
}

export function summarizeLatencies(latenciesMs: number[]): { p50: number; p95: number } {
  const sorted = [...latenciesMs].sort((a, b) => a - b)
  return { p50: percentile(sorted, 0.5), p95: percentile(sorted, 0.95) }
}

// Repeatedly invokes the trivial `ping` command and records each call's
// round-trip time, until `stop()` is called. Meant to run concurrently with
// a line stream so its RTT distribution shows whether webview IPC degrades
// under event/channel load (COD-252 §6).
export function startStatusRttProbe(intervalMs = 50): { stop: () => Promise<{ p50: number; p95: number } | null> } {
  const rtts: number[] = []
  let stopped = false

  async function loop() {
    const { invoke } = await import("@tauri-apps/api/core")
    while (!stopped) {
      const t0 = performance.now()
      try {
        await invoke("ping")
        rtts.push(performance.now() - t0)
      } catch {
        // listener/command unavailable — skip this sample
      }
      await new Promise((r) => setTimeout(r, intervalMs))
    }
  }
  const donePromise = loop()

  return {
    stop: async () => {
      stopped = true
      await donePromise
      return rtts.length > 0 ? summarizeLatencies(rtts) : null
    },
  }
}
