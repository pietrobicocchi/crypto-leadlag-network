# crypto-leadlag-network

[![CI](https://github.com/pietrobicocchi/crypto-leadlag-network/actions/workflows/ci.yml/badge.svg)](https://github.com/pietrobicocchi/crypto-leadlag-network/actions/workflows/ci.yml)

Measuring which crypto instruments move first and which follow, at the
sub-second scale — and testing honestly whether that structure is worth
anything after trading costs.

> **Status:** repository skeleton. Nothing implemented yet.
> Next: `TradeSeries`, then the synthetic generator and the estimator.
> Current scientific state lives in [PROJECT.md](PROJECT.md).

## The problem

Information does not reach all instruments at the same instant. Bitcoin is the
deepest, most-traded instrument, so its price absorbs new information almost
immediately. Smaller coins are quoted by fewer market makers, who need time to
observe and reprice — so a smaller coin's price is briefly stale, reflecting the
world as it was tens or hundreds of milliseconds ago. Measuring that delay, per
pair of instruments, is the core task.

## Why the obvious method is wrong

The naive approach is to chop time into fixed buckets, take the last price in
each, and cross-correlate. Trades do not arrive on a clock, so this breaks in
two ways:

- A thinly traded coin has no trade in most buckets. You carry the last price
  forward and record a return of zero. Measured correlation collapses as buckets
  shrink — the **Epps effect**.
- Worse, the less-frequently-traded asset *appears to lag even when there is no
  true lag*, purely because its returns show up late. The method "discovers"
  that Bitcoin leads everything, for free, and that finding is worthless.

This failure mode is not a footnote. It is the reason the project exists, and
reproducing it deliberately — as a baseline that visibly fails on data with a
known answer — is the first experiment.

## The fix

The **Hayashi–Yoshida** estimator (2005) imposes no clock. It uses each asset's
own trade times, and multiplies a return from asset A by a return from asset B
only when their two time intervals overlap. Summing those products estimates
shared variation with no interpolation and no fabricated zeros.

To find the lag: shift all of B's timestamps by an amount `L`, recompute the
Hayashi–Yoshida correlation, and repeat across a grid of `L`. The value of `L`
at which correlation peaks is the estimated lag.

- Hoffmann, Rosenbaum & Yoshida (2013), *Bernoulli* — estimator and theory.
- Huth & Abergel (2014), *Journal of Empirical Finance* — empirical template.

## What this is and is not

This is a replication of known results with rigorous engineering. It is **not**
a search for a profitable strategy and **not** a claim of novel research. The
expected conclusion is that the effect is real but uneconomic at retail latency
and fees — [an honest negative result is the goal](PROJECT.md#success-metric),
not a problem to engineer around.

## Architecture

A one-way pipeline over immutable artifacts:

```
ingest      ->  downloads, verifies checksums, writes a manifest
normalize   ->  Binance-specific parsing into a canonical form
estimators  ->  pure maths: numpy arrays in, numbers out
study       ->  runs estimators across the universe, caches results
report      ->  figures and the methodology note
```

Three rules keep it one-way, and they are enforced in review, not by a tool:

1. **Nothing in `estimators` may read a file or make a network request.** It
   must be testable with no I/O.
2. **Only `normalize` knows that Binance exists.** Adding a second venue should
   cost one file, not a rewrite.
3. **Notebooks import the library; they never define logic.**

Everything downstream speaks one data structure:

```python
@dataclass(frozen=True)
class TradeSeries:
    symbol: str
    ts_ns: np.ndarray   # int64 nanoseconds, strictly non-decreasing
    price: np.ndarray   # float64
    qty:   np.ndarray   # float64
```

Integer nanoseconds because `float64` loses precision below a microsecond.
Frozen because mutated inputs cause bugs that cannot be reproduced. Parallel
arrays rather than a list of objects because ten million Python objects will not
fit in memory.

## Data

[Binance public archives](https://data.binance.vision) — free, no API key, years
deep. `aggTrades` for USD-M perpetual futures, ~30–40 symbols, 3–6 months.
Archive quirks (mixed headers, a timestamp-unit change mid-2025, missing
periods, `.CHECKSUM` sidecars) are documented in
[PROJECT.md](PROJECT.md#data) and handled by the ingest layer.

## Layout

| Where | What |
| --- | --- |
| `data/binance/` | source archives, verified by SHA-256 (gitignored) |
| `data/canonical/` | parsed `TradeSeries`, ~0.28x the source size (gitignored) |
| `src/leadlag/` | reusable science: the pipeline stages above, plus `run` (provenance) and `plotting` (visual language) |
| `experiments/` | one file per question, `expNNN_name.py` |
| `configs/` | parameters for each experiment, one YAML per experiment |
| `outputs/` | generated results, one directory per experiment (gitignored) |
| `tests/` | tests for code whose silent failure would invalidate a conclusion |
| `notebooks/` | exploration only, never a dependency |
| `report/figures/` | the figures that appear in the report |
| `report/methodology.md` | the argument, and its limitations |

`src/` is what we built, `experiments/` are the questions we asked of it,
`outputs/` is what happened. Experiments import `src/`; `src/` never imports
experiments.

## Install

```bash
make setup     # uv sync --extra dev
```

## Running an experiment

Every experiment starts by calling `start_run()`, which creates its output
directory and writes `metadata.json` — git commit, dirty flag, timestamp, seed,
interpreter and command line. You never have to remember what produced a result:

```
outputs/exp001/
├── config.yaml       the parameters actually used
├── metadata.json     the provenance
└── ...
```

## Report figures

`outputs/` is disposable and messy; `report/figures/` is curated and committed.
A figure crosses that line by being listed in `FIGURES` in
`experiments/figures.py` and nowhere else — that dict is the record of which
experiment produced which figure.

## Commands

```bash
make setup              # install the environment
make test               # pytest
make lint               # ruff check + format --check
make fmt                # ruff format + --fix
make exp EXP=exp001     # run an experiment
make figures            # promote outputs into report/figures/
make clean              # remove caches
```

## Working with an agent

[`AGENTS.md`](AGENTS.md) holds the rules — the pipeline invariants, the figure
conventions, the explore/consolidate modes, and what an agent must ask about
rather than decide. `CLAUDE.md` points there.
[`PROJECT.md`](PROJECT.md) holds the current scientific state.
