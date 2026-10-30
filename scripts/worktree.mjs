#!/usr/bin/env node
// One branch per issue (or short topic), in its own worktree or in the main checkout.
//   node worktree.mjs start <COD-42 | topic>   branch in ../<repo>-<name>, from main
//   node worktree.mjs finish                   from inside a worktree: rebase, check, ff main, push, clean up
//   node worktree.mjs list
//   node worktree.mjs mainline <COD-42 | topic> branch in the main checkout, from main, pushed to origin
//   node worktree.mjs wrap                      from the main checkout: rebase, check, ff main, push, clean up
// The main checkout is the first entry of `git worktree list`. It stays on main unless a mainline branch is open.
// Generalized from aria-studio's scripts/worktree.mjs: detects the package manager and check
// command instead of assuming pnpm + a `check` script, and tolerates a repo with no `origin` yet.
import { execFileSync } from 'node:child_process'
import { cpSync, existsSync, readFileSync } from 'node:fs'
import { basename, dirname, join, resolve } from 'node:path'
import { headMatchesStamp, writeStamp, headSha } from './checkStamp.mjs'

const [command, arg] = process.argv.slice(2)

const run = (cmd, args, cwd) => execFileSync(cmd, args, { cwd, stdio: 'inherit' })
const read = (args, cwd) => execFileSync('git', args, { cwd, encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] }).trim()
const fail = (message) => {
  process.stderr.write(`wt: ${message}\n`)
  process.exit(1)
}
const step = (label, fn, hint) => {
  console.log(`\nwt: ${label}`)
  try {
    fn()
  } catch {
    fail(`${label} failed. ${hint}`)
  }
}

const mainRoot = read(['worktree', 'list', '--porcelain']).split('\n')[0].replace(/^worktree /, '')
const branchOf = (cwd) => read(['rev-parse', '--abbrev-ref', 'HEAD'], cwd)
const isDirty = (cwd, untracked) =>
  read(['status', '--porcelain', `--untracked-files=${untracked ? 'all' : 'no'}`], cwd) !== ''
const refExists = (ref) => {
  try {
    read(['rev-parse', '--verify', '--quiet', ref], mainRoot)
    return true
  } catch {
    return false
  }
}
const branchExists = (name) => refExists(`refs/heads/${name}`)
const hasOrigin = (() => {
  try {
    read(['remote', 'get-url', 'origin'], mainRoot)
    return true
  } catch {
    return false
  }
})()

// --- package manager + check-command detection (generalizes the aria-studio assumption of pnpm + `check`) ---
const packageJsonPath = (cwd) => join(cwd, 'package.json')
const readPackageJson = (cwd) => {
  const file = packageJsonPath(cwd)
  if (!existsSync(file)) return null
  try {
    return JSON.parse(readFileSync(file, 'utf8'))
  } catch {
    return null
  }
}
const detectPm = (cwd) => {
  if (existsSync(join(cwd, 'pnpm-lock.yaml'))) return 'pnpm'
  if (existsSync(join(cwd, 'yarn.lock'))) return 'yarn'
  if (existsSync(join(cwd, 'package-lock.json'))) return 'npm'
  return existsSync(packageJsonPath(cwd)) ? 'npm' : null
}
const installCommand = (pm) => {
  if (pm === 'pnpm') return ['pnpm', ['install', '--frozen-lockfile', '--prefer-offline']]
  if (pm === 'yarn') return ['yarn', ['install', '--frozen-lockfile']]
  return ['npm', ['ci']]
}
// Prefer a single `check` script (lint+typecheck+test+build, as in aria-studio); otherwise
// run whichever of typecheck/test/build scripts the project actually defines; otherwise skip.
const checkPlan = (cwd) => {
  const pm = detectPm(cwd)
  if (!pm) return null
  const pkg = readPackageJson(cwd)
  const scripts = pkg?.scripts ?? {}
  if (scripts.check) return { pm, runs: [['check']] }
  const fallback = ['typecheck', 'test', 'build'].filter((name) => scripts[name])
  if (fallback.length > 0) return { pm, runs: fallback.map((name) => [name]) }
  return null
}
const runCheck = (cwd) => {
  const plan = checkPlan(cwd)
  if (!plan) {
    console.log('\nwt: no check/typecheck/test/build script found in package.json; skipping checks')
    return
  }
  for (const args of plan.runs) {
    step(`${plan.pm} run ${args[0]}`, () => run(plan.pm, ['run', ...args], cwd), 'Fix the failures, commit, then run this command again.')
  }
}
// The branch was just rebased onto main, whose tip passed a check, so re-running
// on the same commit would be redundant.
const checkUnlessStamped = (cwd) => {
  if (headMatchesStamp(cwd)) {
    console.log(`\nwt: checks already passed for ${headSha(cwd)}`)
    return
  }
  runCheck(cwd)
  if (checkPlan(cwd)) writeStamp(cwd, headSha(cwd))
}

const mainOnMain = () => {
  if (branchOf(mainRoot) !== 'main')
    fail(`the main checkout (${mainRoot}) is not on main. If a mainline branch is open there, finish it with \`wrap\` first.`)
}
const branchName = (usage) => {
  if (!arg) fail(`usage: ${usage}`)
  const name = arg.toLowerCase()
  if (!/^[a-z0-9][a-z0-9._-]*$/.test(name)) fail(`"${arg}" is not a usable branch name`)
  return name
}
const pullMain = () => {
  if (!hasOrigin) return
  try {
    run('git', ['pull', '--ff-only', 'origin', 'main'], mainRoot)
  } catch {
    console.warn('wt: could not update main from origin; branching from the local main')
  }
}
const pushStep = (label, args, cwd, hint) => {
  if (!hasOrigin) {
    console.log(`\nwt: ${label} skipped (no origin remote configured yet)`)
    return
  }
  step(label, () => run('git', args, cwd), hint)
}

function start() {
  const name = branchName('node worktree.mjs start <COD-42 | short-topic>')
  const dir = join(dirname(mainRoot), `${basename(mainRoot)}-${name}`)

  if (read(['worktree', 'list', '--porcelain']).split('\n').includes(`worktree ${dir}`)) {
    console.log(`wt: ${name} already has a worktree:\n${dir}`)
    return
  }
  if (existsSync(dir)) fail(`${dir} exists but is not a worktree. Move it away first.`)
  mainOnMain()

  pullMain()
  const reuse = branchExists(name)
  step(
    reuse ? `add worktree for existing branch ${name}` : `create branch ${name} from main`,
    () => run('git', ['worktree', 'add', dir, ...(reuse ? [name] : ['-b', name, 'main'])], mainRoot),
    'See the git output above.',
  )
  const pm = detectPm(mainRoot)
  if (pm) {
    const [cmd, args] = installCommand(pm)
    step('install dependencies', () => run(cmd, args, dir), `Run it by hand in ${dir}.`)
  }

  // Gitignored local files a fresh checkout lacks. Extend this list per project if needed.
  for (const file of ['.env', '.env.local', '.claude/settings.local.json']) {
    if (existsSync(join(mainRoot, file))) cpSync(join(mainRoot, file), join(dir, file), { recursive: true })
  }
  console.log(`\nwt: ready. Branch ${name}, worktree:\n${dir}`)
}

function finish() {
  const root = read(['rev-parse', '--show-toplevel'])
  if (resolve(root) === resolve(mainRoot)) fail('run finish from inside the issue worktree, not the main checkout')
  const branch = branchOf(root)
  if (branch === 'HEAD' || branch === 'main') fail(`the worktree is on ${branch}, not an issue branch`)
  if (isDirty(root, true)) fail('the worktree has uncommitted or untracked changes. Commit or remove them first.')
  mainOnMain()
  if (isDirty(mainRoot, false)) fail(`the main checkout (${mainRoot}) has uncommitted changes`)

  if (hasOrigin)
    step('update main from origin', () => run('git', ['pull', '--ff-only', 'origin', 'main'], mainRoot), 'Fix main first.')
  step(
    `rebase ${branch} onto main`,
    () => run('git', ['rebase', 'main'], root),
    'Resolve the conflicts, run `git rebase --continue`, then run finish again.',
  )
  checkUnlessStamped(root)
  step(
    `fast-forward main to ${branch}`,
    () => run('git', ['merge', '--ff-only', branch], mainRoot),
    'main moved on; run finish again.',
  )
  pushStep('push main', ['push', 'origin', 'main'], mainRoot, 'main is merged locally; push it by hand.')
  step(
    'remove the worktree',
    () => run('git', ['worktree', 'remove', root], mainRoot),
    `main is merged${hasOrigin ? ' and pushed' : ''}; remove ${root} by hand.`,
  )
  step(`delete branch ${branch}`, () => run('git', ['branch', '-d', branch], mainRoot), 'Delete it by hand.')
  console.log(`\nwt: ${branch} is on main${hasOrigin ? ' and pushed' : ' (no origin yet)'}. Continue in:\n${mainRoot}`)
}

function mainline() {
  const name = branchName('node worktree.mjs mainline <COD-42 | short-topic>')
  const root = read(['rev-parse', '--show-toplevel'])
  if (resolve(root) !== resolve(mainRoot)) fail(`run mainline from the main checkout (${mainRoot}), not a worktree`)
  const current = branchOf(mainRoot)
  if (current === name) {
    console.log(`wt: already on ${name} in ${mainRoot}`)
    return
  }
  if (current !== 'main') fail(`the main checkout is on ${current}. Finish it with \`wrap\` first.`)
  if (isDirty(mainRoot, false)) fail('the main checkout has uncommitted changes. Commit or stash them first.')

  pullMain()
  const reuse = branchExists(name)
  step(
    reuse ? `switch to existing branch ${name}` : `create branch ${name} from main`,
    () => run('git', ['switch', ...(reuse ? [name] : ['-c', name, 'main'])], mainRoot),
    'See the git output above.',
  )
  if (!hasOrigin) {
    console.log(`\nwt: ready. Branch ${name} in:\n${mainRoot}\nwt: no origin remote yet; skipping push`)
    return
  }
  let tracked = false
  try {
    read(['rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{u}'], mainRoot)
    tracked = true
  } catch {
    // no upstream yet
  }
  if (!tracked)
    step(
      `push ${name} to origin`,
      () => run('git', ['push', '-u', 'origin', name], mainRoot),
      `The branch is ready locally; push it by hand with \`git push -u origin ${name}\`.`,
    )
  console.log(`\nwt: ready. Branch ${name} in:\n${mainRoot}`)
}

function wrap() {
  const root = read(['rev-parse', '--show-toplevel'])
  if (resolve(root) !== resolve(mainRoot)) fail('this is a worktree; finish it with `finish` instead')
  const branch = branchOf(root)
  if (branch === 'HEAD' || branch === 'main') fail(`the main checkout is on ${branch}, not an issue branch`)
  if (isDirty(root, true)) fail('the checkout has uncommitted or untracked changes. Commit or remove them first.')

  if (hasOrigin)
    step(
      'update main from origin',
      () => run('git', ['fetch', 'origin', 'main:main'], root),
      'Local main has commits origin lacks, or origin is unreachable. Fix main first.',
    )
  step(
    `rebase ${branch} onto main`,
    () => run('git', ['rebase', 'main'], root),
    'Resolve the conflicts, run `git rebase --continue`, then run wrap again.',
  )
  checkUnlessStamped(root)
  step('switch to main', () => run('git', ['switch', 'main'], root), 'See the git output above.')
  step(
    `fast-forward main to ${branch}`,
    () => run('git', ['merge', '--ff-only', branch], root),
    `main moved on; run \`git switch ${branch}\` and wrap again.`,
  )
  if (!hasOrigin) {
    step(`delete branch ${branch}`, () => run('git', ['branch', '-D', branch], root), 'Delete it by hand.')
    console.log(`\nwt: ${branch} is on main (no origin yet). The checkout is back on main.`)
    return
  }
  // One push moves main and deletes the branch's origin copy, known from the
  // tracking ref mainline set up. --atomic: neither happens without the other.
  const remote = refExists(`refs/remotes/origin/${branch}`)
  step(
    remote ? `push main and delete ${branch} on origin` : 'push main',
    () => run('git', ['push', '--atomic', 'origin', 'main', ...(remote ? [`:${branch}`] : [])], root),
    remote
      ? `main is merged locally; push it by hand, then \`git push origin --delete ${branch}\` if origin still has it.`
      : 'main is merged locally; push it by hand.',
  )
  // -D: main now contains the branch, but its stale origin copy would make -d refuse.
  step(`delete branch ${branch}`, () => run('git', ['branch', '-D', branch], root), 'Delete it by hand.')
  console.log(`\nwt: ${branch} is on main and pushed. The checkout is back on main.`)
}

if (command === 'start') start()
else if (command === 'finish') finish()
else if (command === 'list') run('git', ['worktree', 'list'])
else if (command === 'mainline') mainline()
else if (command === 'wrap') wrap()
else fail('usage: worktree.mjs start <COD-42 | topic> | finish | list | mainline <COD-42 | topic> | wrap')
