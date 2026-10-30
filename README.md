# LaserSim

Tauri + React + TypeScript shell for laser burn-simulation spikes (Linear project "Laser Sim").

[COD-251](https://linear.app/codeagents/issue/COD-251/spike-tauri-webgl2-burn-sim-performance-test-app): spike proving (or disproving) that a Tauri app can run the burn simulation at full resolution in the webview (WebGL2) while keeping a GRBL TCP emulator in the native (Rust) shell. See the issue for the pass/fail criteria.

## Workflow

Branches follow the `issue-branch-workflow` pattern (`scripts/worktree.mjs`): one branch per issue, no PRs, fast-forwarded into `main` locally.

- `npm run mainline COD-42` / `npm run wrap` — single branch in this checkout.
- `npm run wt:start COD-42` / `npm run wt:finish` / `npm run wt:list` — branch in its own git worktree, for parallel work.

## Dev

```bash
npm install
npm run tauri dev
```

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
