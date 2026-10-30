// A clean check run records HEAD here. finish/wrap (and an optional pre-push
// hook) skip the suite when the commit being pushed is that sha. The file
// lives in the common git dir, so a worktree and the main checkout share it.
import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

export const DELETE_SHA = '0'.repeat(40)
const STAMP_NAME = 'issue-branch-check-stamp'
const MAIN_REF = 'refs/heads/main'

const git = (args, cwd) => execFileSync('git', args, { cwd, encoding: 'utf8' }).trim()

export const stampFile = (cwd = process.cwd()) => {
  const dir = git(['rev-parse', '--git-common-dir'], cwd)
  return join(resolve(cwd, dir), STAMP_NAME)
}

export const readStamp = (cwd = process.cwd()) => {
  const file = stampFile(cwd)
  if (!existsSync(file)) return null
  const sha = readFileSync(file, 'utf8').trim().toLowerCase()
  return /^[0-9a-f]{40}$/.test(sha) ? sha : null
}

export const writeStamp = (cwd, sha) => {
  writeFileSync(stampFile(cwd), `${sha.toLowerCase()}\n`)
}

export const clearStamp = (cwd = process.cwd()) => {
  const file = stampFile(cwd)
  if (existsSync(file)) unlinkSync(file)
}

export const isClean = (cwd = process.cwd()) =>
  git(['status', '--porcelain', '--untracked-files=all'], cwd) === ''

export const headSha = (cwd = process.cwd()) => git(['rev-parse', 'HEAD'], cwd).toLowerCase()

export const headMatchesStamp = (cwd = process.cwd()) => {
  const stamp = readStamp(cwd)
  return stamp !== null && stamp === headSha(cwd)
}

// Pre-push lines are `<local ref> <local sha> <remote ref> <remote sha>`.
// Only updates of main need the stamped commit. Deletes and other branches
// (work in progress) reach main only through finish or wrap, which check.
export const mainUpdates = (lines) => {
  const shas = []
  for (const line of lines) {
    const [, localSha, remoteRef] = line.trim().split(/\s+/)
    if (remoteRef !== MAIN_REF || !localSha || localSha.toLowerCase() === DELETE_SHA) continue
    shas.push(localSha.toLowerCase())
  }
  return shas
}

export const refsNeedCheck = (stamp, lines) => {
  const shas = mainUpdates(lines)
  if (shas.length === 0) return false
  if (!stamp) return true
  return shas.some((sha) => sha !== stamp.toLowerCase())
}
