# AGENTS.md

Rules for any coding agent working in this repository. Loaded every session —
keep it under ~150 lines, and keep it true.

## What this is

Research code that has to become publication-quality code.

Optimise, in this order: **correctness → readability → reproducibility → simplicity.**

Do not introduce abstractions for hypothetical future requirements.

## Structure

```
src/leadlag/        reusable science — what we built and discovered
experiments/        entrypoints — the questions we asked of it
configs/            parameters only, never logic
outputs/            generated results, one directory per experiment (gitignored)
tests/              tests for scientifically dangerous code
notebooks/          exploration only
paper/figures/      the figures that appear in the paper
paper/outline.md    the scientific narrative
```

## The invariant

**`experiments/` may import `src/`. `src/` must never import `experiments/`.**

- Reusable logic belongs in `src/`; experiment-specific logic stays in that experiment's file.
- A notebook must never hold the only implementation of anything.
- No reproduction path may depend on running notebook cells in a particular order.

## Before creating a file

Ask whether it belongs in a file that already exists. Prefer editing.

Do not create new markdown files, helper modules, `utils.py`, `common.py`, or
abstraction layers unless there is a clear structural benefit. **Never create
documentation files.** Update `README.md`, `AGENTS.md` or `PROJECT.md` instead:
this repository documents what is true now, and git remembers what happened.

## Code preferences

Prefer modifying a file over creating one.
Prefer a function over a class.
Prefer a class over a framework.
Prefer duplication over a premature abstraction.
Delete obsolete code rather than preserve compatibility, unless compatibility is
explicitly required.

Three duplicated lines are usually cheaper than an abstraction in research code.
Functions should name concepts that can be named scientifically.

## Experiments

- One experiment answers one identifiable question, stated in its docstring.
- `expNNN_short_name.py` writes only to `outputs/expNNN/`.
- Call `start_run()` from `src/leadlag/run.py` first: it records the config,
  git commit, dirty flag, seed and command line that produced the results.
- Never edit a generated output by hand.
- Run with `make exp EXP=exp007`.

## Figures

- Every figure goes through `src/leadlag/plotting.py`. Never set fonts,
  sizes or colours ad hoc inside an experiment.
- Colours and display names come from `METHOD_COLORS` and `METHOD_LABELS`, and
  must be identical in every figure.
- Vector output (PDF) for plots.
- Axes carry units where applicable. Sentence case. No titles on paper figures
  unless scientifically necessary. No redundant legends.
- Never truncate an axis in a way that distorts a comparison.
- An uncertainty band must state what quantity it represents.
- A figure becomes a paper figure only by being added to `FIGURES` in
  `experiments/figures.py`, then `make figures`.

## Tests

Risk-weighted, not coverage-weighted. Test where a silent error would invalidate
a scientific conclusion: metric implementations, mathematical identities and
invariants, array shapes, preprocessing, data splitting, determinism,
serialisation, numerically delicate routines.

Do not test experiment orchestration, plotting boilerplate, trivial wrappers or
CLI arguments. Never add a test to raise coverage.

## Dependencies

Prefer the standard library and what is already installed. Ask before adding a
substantial dependency. All configuration lives in `pyproject.toml`.

## Two modes

Say which mode you are in when it is ambiguous.

**EXPLORE** — the question is open. Work in `experiments/`. Duplication, debug
prints and throwaway code are fine. Do not refactor `src/`.

**CONSOLIDATE** — something worked. Move the reusable implementation into `src/`,
delete the exploratory copy, point the experiment at the canonical version, and
add a test for the invariant that matters.

## Decide freely / ask first

Without asking: implement functions, refactor locally, write targeted tests, run
experiments, fix lint, produce plots, inspect outputs, simplify code.

Stop and ask before: changing the scientific question; changing a metric
definition; changing dataset splits or statistical methodology; adding a major
dependency; introducing an architectural layer; changing a public interface in
`src/`; deleting experiment results; making an assumption that affects a
conclusion.

## Finishing a substantial change

Run `make test` and `make lint`, and execute any experiment entrypoint you
changed. Then report, briefly:

**Changed** — one line per file: what and why
**Architecture** — why the new code lives where it lives
**Data flow** — the path from data to result, if it changed
**Assumptions** — anything you decided that I should verify
**Run** — the exact command to reproduce it

That report is the deliverable. My job is to hold the mental model of the
system; yours is to keep it accurate.
