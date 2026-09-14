# crypto-leadlag-network

<One sentence: the question this project answers.>

## Install

```bash
make setup     # uv sync --extra dev
```

## Reproduce the main result

```bash
make exp EXP=exp001     # -> outputs/exp001/
make figures            # -> report/figures/
```

## Layout

| Where | What |
| --- | --- |
| `src/leadlag/` | reusable science: `data`, `methods`, `metrics`, `plotting`, `run` |
| `experiments/` | one file per question, `expNNN_name.py` |
| `configs/` | parameters for each experiment, one YAML per experiment |
| `outputs/` | generated results, one directory per experiment (gitignored) |
| `tests/` | tests for code whose silent failure would invalidate a conclusion |
| `notebooks/` | exploration only, never a dependency |
| `report/figures/` | the figures that appear in the report |
| `report/methodology.md` | the scientific narrative |

Three layers, one direction: `src/` is what we built, `experiments/` are the
questions we asked of it, `outputs/` is what happened. Experiments import `src/`;
`src/` never imports experiments.

## Running an experiment

Every experiment starts by calling `start_run()`, which creates its output
directory and writes `metadata.json` — git commit, dirty flag, timestamp, seed,
interpreter and command line. You never have to remember what produced a result:

```
outputs/exp001/
├── config.yaml       the parameters actually used
├── metadata.json     the provenance
├── metrics.csv
└── alpha_sweep.pdf
```

## Report figures

`outputs/` is disposable and messy; `report/figures/` is curated and committed. A
figure crosses that line by being listed in `FIGURES` in `experiments/figures.py`
and nowhere else — that dict is the record of which experiment produced which
figure.

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

[`AGENTS.md`](AGENTS.md) holds the rules — the import invariant, the ban on new
files and documentation, the figure conventions, the explore/consolidate modes,
and what the agent must ask about rather than decide. `CLAUDE.md` points there.
[`PROJECT.md`](PROJECT.md) holds the current scientific state.
