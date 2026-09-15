# AGENTS.md

Rules for any coding agent working in this repository. Loaded every session —
keep it under ~150 lines, and keep it true.

## What this is

Measuring lead–lag structure between crypto instruments at the sub-second scale,
then testing honestly whether it survives trading costs. Research code that has
to become publication-quality code.

Optimise, in this order: **correctness → readability → reproducibility →
simplicity.** Do not introduce abstractions for hypothetical requirements.

## How to work with me

I am learning to code. **Optimise for my understanding, not for speed.**

- Explain the plan in plain English before writing code. Wait for my go.
- One file per change.
- No function body until the signature, docstring and a test exist.
- When I ask how to do something, give me 2–3 options with tradeoffs rather
  than choosing for me.
- Explain any Python feature a beginner would not know.
- Never add a dependency without justifying it.

## Structure

```
data/             downloaded source archives (gitignored, large)
src/leadlag/      reusable science, one module per pipeline stage
experiments/      entrypoints — the questions we asked of it
configs/          parameters only, never logic
outputs/          results, one directory per experiment (gitignored)
tests/            tests for scientifically dangerous code
notebooks/        exploration only
report/           figures/ and methodology.md
```

One-way over immutable artifacts. A module appears when implemented, not before:

    ingest -> normalize -> estimators -> study -> report

## The invariants

**`experiments/` may import `src/`. `src/` must never import `experiments/`.**

**Nothing in `estimators` may read a file or make a network request.** Pure
maths: numpy arrays in, numbers out. It must be testable with no I/O. If a test
for an estimator needs a fixture file, the design is wrong.

**Only `ingest` and `normalize` know that Binance exists**, and they know
different halves of it. `ingest` knows where bytes come from: the URL layout
and the `.CHECKSUM` sidecars. `normalize` knows what is inside them: columns,
header rows, timestamp units, every other quirk. Neither knowledge may leak
past those two files, so adding a venue costs two files and not a rewrite.

**Notebooks import the library; they never define logic.** No reproduction path
may depend on running notebook cells in a particular order.

## The central data structure

Everything downstream speaks `TradeSeries` (`leadlag.types`):

```python
@dataclass(frozen=True)
class TradeSeries:
    symbol: str
    ts_ns: np.ndarray   # int64 nanoseconds, strictly non-decreasing
    price: np.ndarray   # float64
    qty:   np.ndarray   # float64
```

Integer nanoseconds, never float seconds: `float64` loses precision below a
microsecond and this project measures milliseconds. Frozen: mutated inputs cause
bugs that cannot be reproduced. Parallel arrays, not objects: ten million Python
objects will not fit in memory. Timestamp units are **detected, never assumed** —
the Binance archive changed from milliseconds to microseconds mid-2025.

## Scientific honesty

The expected conclusion is that the effect is real but uneconomic. **An honest
negative result is the goal, not a problem to engineer around.** Never relax a
cost, latency or fee assumption to improve a result. If a result looks good,
suspect a lookahead bug first. Report missing data, never silently fill it — a
gap in the archive is normal and belongs in the manifest.

## Experiments

- One experiment answers one identifiable question, stated in its docstring.
- `expNNN_short_name.py` writes only to `outputs/expNNN/`. Run it with
  `make exp EXP=exp007`. Never edit a generated output by hand.
- Call `start_run()` from `src/leadlag/run.py` first: it records the config, git
  commit, dirty flag, seed and command line that produced the results.

## Figures

- Every figure goes through `src/leadlag/plotting.py`. Never set fonts, sizes or
  colours ad hoc. Colours and names come from `METHOD_COLORS` / `METHOD_LABELS`
  and are identical everywhere: the gridded baseline is the same grey line in
  every figure it fails in.
- Vector (PDF). Axes carry units. Sentence case. No redundant legends. Never
  truncate an axis in a way that distorts a comparison. An uncertainty band must
  state what quantity it represents.
- A figure becomes a report figure by being added to `FIGURES` in
  `experiments/figures.py`, then `make figures`. Nowhere else.

## Tests

Risk-weighted, not coverage-weighted. Test where a silent error would invalidate
a conclusion: estimators against known analytic answers, mathematical identities
and invariants, array shapes and dtypes, timestamp-unit handling, determinism.

The test that matters most: on synthetic data where B is a known delayed copy of
A with independent Poisson arrivals, the estimator recovers the known lag — and
still does when A trades 10x more often than B, where the gridded baseline
visibly fails. That runs in CI.

Do not test orchestration, plotting boilerplate or trivial wrappers. Never add a
test to raise coverage.

## Before creating a file

Ask whether it belongs in a file that already exists. Prefer editing. Do not
create `utils.py`, `common.py` or abstraction layers. **Never create
documentation files** — update `README.md`, `AGENTS.md` or `PROJECT.md`: this
repository documents what is true now, and git remembers what happened.

Prefer a function over a class; a class over a framework; duplication over a
premature abstraction. Delete obsolete code rather than preserve compatibility.
Functions should name concepts that can be named scientifically.

## Two modes

Say which mode you are in when it is ambiguous.

**EXPLORE** — the question is open. Work in `experiments/`; duplication and
throwaway code are fine. Do not refactor `src/`.

**CONSOLIDATE** — something worked. Move it into `src/`, delete the exploratory
copy, point the experiment at the canonical version, add a test for the
invariant that matters.

## Decide freely / ask first

Without asking: implement an agreed function, refactor locally, write targeted
tests, run experiments, fix lint, produce plots, inspect outputs, simplify code.

Stop and ask before: changing the scientific question; changing an estimator or
metric definition; changing the lag grid, universe or window length; adding a
dependency; introducing an architectural layer; changing a public interface in
`src/`; deleting experiment results; making an assumption that affects a
conclusion.

## Finishing a substantial change

Run `make test` and `make lint`, and execute any experiment entrypoint you
changed. Then report briefly:

**Changed** — one line per file: what and why
**Architecture** — why the new code lives where it lives
**Data flow** — the path from data to result, if it changed
**Assumptions** — anything you decided that I should verify
**Run** — the exact command to reproduce it

That report is the deliverable. My job is to hold the mental model of the
system; yours is to keep it accurate.
