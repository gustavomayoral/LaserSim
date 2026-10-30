#!/usr/bin/env node
// COD-251 "also check": streams a synthetic G-code job over TCP to the running
// Tauri app's gcode_listener (src-tauri/src/gcode_listener.rs) as fast as
// possible, so the app's reported lines/sec and latency show whether the
// native-to-webview IPC hop is a bottleneck. Run the app first (npm run tauri
// dev), then: node scripts/send-gcode.mjs [lineCount] [port]
import { connect } from "node:net"

const lineCount = Number(process.argv[2] ?? 50000)
const port = Number(process.argv[3] ?? 8090)

const socket = connect(port, "127.0.0.1", () => {
  console.log(`send-gcode: connected to 127.0.0.1:${port}, sending ${lineCount} lines`)
  const start = performance.now()
  let x = 0
  let y = 0
  for (let i = 0; i < lineCount; i++) {
    x = (x + 0.1) % 100
    y = (y + 0.0001) % 100
    socket.write(`G1 X${x.toFixed(4)} Y${y.toFixed(4)} F500\n`)
  }
  socket.end(() => {
    const elapsedMs = performance.now() - start
    console.log(
      `send-gcode: wrote ${lineCount} lines in ${elapsedMs.toFixed(1)}ms ` +
        `(${((lineCount / elapsedMs) * 1000).toFixed(0)} lines/sec sent)`,
    )
  })
})

socket.on("error", (err) => {
  console.error(`send-gcode: ${err.message}. Is the app running (npm run tauri dev)?`)
  process.exit(1)
})
