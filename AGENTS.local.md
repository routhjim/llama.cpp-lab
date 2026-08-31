# AGENTS.local.md - rules for THIS repository

**Read this before `AGENTS.md`.** This file governs work in this repository. `AGENTS.md` is
upstream's file, vendored with the source, and it governs work destined for
`ggml-org/llama.cpp`. The two say different things on purpose.

## What this repository is

`routhjim/llama.cpp-lab` is a personal working copy of llama.cpp. It is **not**
`ggml-org/llama.cpp` and it is **not a GitHub fork of it** - `isFork` is false, there is no
parent repository, and no upstream bot watches it. It exists to hold experimental work
(MTP for qwen4exp, MoE expert tiering, Vulkan batch tuning) as reviewable topical branches
rather than one unreadable WIP commit.

Nothing here is offered to upstream. If a change ever should go upstream, a human opens
that PR from a real fork, under upstream's rules, and owns it.

## What is permitted here

- **Commits, branches and pushes** to `origin` (this repo).
- **Writing pull-request descriptions**, and opening PRs with `gh pr create --repo
  routhjim/llama.cpp-lab`.
- Use `Assisted-by: <model name>` when an assistant wrote a commit. Do not use
  `Co-authored-by:` for an AI - that trailer is for humans whose code was actually derived
  from (e.g. the `Co-authored-by: JJJYmmm` on the MTP commit, which is correct and must be
  preserved).

## What is forbidden here, without exception

- **Never push to `upstream`.** Its push URL is set to the literal string `DISABLED` so the
  attempt fails loudly. Do not "fix" that.
- **Never open a pull request against `ggml-org/llama.cpp`.** Always pass
  `--repo routhjim/llama.cpp-lab` explicitly; never rely on `gh` inferring the base.
- **Never comment on, or reply to, anything in an upstream repository.**

Four independent guards enforce this: the disabled `upstream` push URL, the absence of a
`fork` remote (gh resolves a fork's base repo to its *parent*, which is the real hazard),
`remote.origin.gh-resolved = base`, and the explicit `--repo` flag. If you find any of them
missing, restore it before doing anything else.

## Branch layout

- **`master`** - a pristine mirror of `upstream/master`. Only ever fast-forwarded. Never a
  PR base, never merged into.
- **`main`** - the default branch and the base for every PR here.
- **`pr/<topic>`** - one branch per PR. A PR whose work depends on another opens with
  `--base pr/<other>`; GitHub retargets it to `main` when the parent merges.
- **`archive/*` tags** - preserved dead ends. Do not delete; some carry attribution
  provenance.

## Upstream sync

```sh
git fetch upstream master --prune
git push origin +upstream/master:master        # advance the mirror
git switch main && git merge --no-ff upstream/master
git switch pr/<topic> && git rebase main && git push --force-with-lease
```

## Verification is local

GitHub Actions is disabled on this repository, deliberately - upstream's workflow matrix is
~90 jobs and provides nothing useful here. That means **no PR gets automated checks**. Paste
the actual command and its real output into the PR body; that is the only record that a
check was ever run. Do not describe a test you did not run.
