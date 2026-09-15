# PROJECT.md

The living scientific state of this project: what we are investigating, and what
we currently know. **Update this file in place — never append to it as a log.**
Leave a prompt in any section that is not yet true.

## Question

Information does not reach every crypto instrument at the same instant. Bitcoin
is the deepest and most heavily traded, so its price absorbs news almost
immediately; a thinner altcoin is quoted by fewer market makers, who need time
to observe and reprice, leaving its price briefly stale. **Which instruments
lead, which follow, by how many milliseconds — and does that structure survive
realistic trading costs?** This is a replication of known results
(Hoffmann–Rosenbaum–Yoshida 2013; Huth–Abergel 2014) with rigorous engineering,
not a claim of novel research.

## Hypothesis

A genuine, non-zero lead–lag structure exists at the sub-second scale, with
leadership ordered roughly by liquidity, and it is **not economically
exploitable** at retail latency and fee levels.

Falsified if: the Hayashi–Yoshida lag estimates are indistinguishable from zero
once bootstrap confidence intervals and multiple-testing correction are applied;
or if leadership rank is unstable across rolling windows; or if net Sharpe stays
positive across the whole plausible latency/fee range, which would more likely
indicate a bug in the simulator than an edge.

## Success metric

Two gates, in order. The first decides whether anything downstream is believable.

**Gate 1 — the estimator is correct. PASSING.** On synthetic data where B is a known
delayed copy of A, with independent Poisson arrivals and A trading 10x more
often than B, Hayashi–Yoshida must recover the true lag to within one grid step,
while the naive gridded baseline visibly fails. This runs in CI. Until it
passes, no result on real data means anything.

**Gate 2 — the headline number.** Net Sharpe of the event-time strategy as a
function of assumed round-trip latency and fee level.

**A net Sharpe indistinguishable from zero at retail latency and fees is a
successful outcome, not a failed one.** The deliverable is a trustworthy answer,
not a positive one. If a later version of me is tempted to loosen a cost
assumption so this curve clears zero, that is the moment this project has
failed.

## Data

| Dataset | Role | Location | Splits |
| --- | --- | --- | --- |
| Binance `aggTrades`, USD-M perpetual futures | canonical | `data.binance.vision` (public archive, no API key) | ~30–40 symbols, 3–6 months; rolling windows, no train/test split — this is estimation, not prediction |
| Synthetic delayed-copy series | validation | generated in `leadlag.synthetic` | n/a — ground truth is known by construction |

Known traps in the archive, all of which the ingest layer must handle rather
than crash on:

- Some CSVs carry a header row and some do not.
- The timestamp unit changed from milliseconds to microseconds mid-2025 in some
  files. Units must be detected, never assumed.
- Every archive has a `.CHECKSUM` sidecar; SHA-256 is verified on download.
- Some periods were never published. A missing file is normal and must be
  **reported in the manifest, not raised**.

**Decision: simultaneous trades are collapsed at construction.** `TradeSeries`
holds one row per distinct timestamp — last price wins, quantities summed — so
`ts_ns` is strictly increasing. This is a data decision, not a detail. Hayashi–
Yoshida works on intervals between consecutive observations; two trades sharing
a timestamp would form a zero-length interval that overlaps nothing, silently
discarding the price change across it. Collapsing solves this once, in one
place, rather than requiring every estimator to handle degenerate intervals.

## Baselines

| Baseline | Why it is the right comparison | Status |
| --- | --- | --- |
| Naive gridded cross-correlation | It is what you get by chopping time into fixed buckets and forward-filling. It is the standard wrong answer: it manufactures a lead-lag finding from trade-rate asymmetry alone, so it "discovers" that Bitcoin leads everything for free. Showing it fail on synthetic data with a known answer is the argument for Hayashi–Yoshida. | **implemented**, and confirmed to fail as predicted |
| Shuffled-timestamp null | Destroys cross-asset timing while preserving each series' marginal distribution. Any lag surviving this is an artefact. | not implemented |
| Hayashi–Yoshida | The estimator under test, not a baseline — listed here because the comparison against the gridded method is the argument. | **implemented**, Gate 1 passing |

## Experiments

| ID | Question | Status | Result |
| --- | --- | --- | --- |
| exp001 | Does Hayashi–Yoshida recover a known lag on asynchronous synthetic data where the gridded baseline fails? | **done** | `outputs/exp001/`, `report/figures/fig1_synthetic_recovery.pdf` |
| exp002 | What is the pairwise lag matrix across the universe, with bootstrap CIs and Benjamini–Hochberg correction over ~600 tests? | not started | — |
| exp003 | Is the directed lead–lag network stable across rolling windows, and does leadership rank track liquidity? | not started | — |
| exp004 | What is net Sharpe as a function of round-trip latency and fee level? | not started | — |

## What we have learned

- Nothing scientific yet. `TradeSeries` exists and its contract is enforced by
  20 tests; simultaneous trades are rejected at construction rather than
  silently forming zero-length intervals.
- The synthetic generator produces a known lag with the correct sign: a scan of
  candidate shifts bottoms out at the injected lag, 18.7x below the wrong-sign
  shift. CI runs that check on every push.
- **The gridded baseline fabricates a lead-lag finding from trade-rate
  asymmetry alone.** With a true lag of exactly zero and A trading 10x more
  often than B, the Huth–Abergel lead-lag ratio reaches 64–180 (symmetric
  control: 0.7–1.4), while peak correlation falls from 0.95 to 0.60. Critically
  the peak itself stays at zero — the artefact is invisible to anyone reading
  only the argmax, which is the usual summary. Established in
  `tests/test_estimators.py`; awaiting exp001 for the figure.
- **Hayashi–Yoshida does not make that mistake, and still finds real lags.** On
  identical data (true lag zero, 10:1 imbalance) it reports a lead-lag ratio of
  0.96–1.03 against the baseline's 197–387, and it recovers injected lags of
  100 ms and 250 ms exactly. Gate 1 passes.
- **The fabricated finding is not monotone in trade-rate imbalance.** Median
  gridded lead-lag ratio over ten seeds: 1.2 at 1:1, 6.4 at 2:1, 68 at 5:1, 106
  at 10:1, then *down* to 70 at 20:1 and 46 at 50:1. Hayashi–Yoshida stays at
  0.84–0.98 throughout. The decline past 10:1 was not predicted; the plausible
  cause is that B becomes so sparse that every correlation weakens and the
  ratio compresses, but that is a hypothesis, not a measurement. It does not
  affect Gate 1.
- **Two of the archive's documented traps behave differently than expected.**
  Header rows: real — USD-M `aggTrades` files from 2020 and 2021 have none,
  2023 onward do. Microsecond timestamps: *not observed*. Every sampled file
  from 2020-01 to 2026-09, including the mid-2025 window, is 13-digit
  milliseconds. Seven samples across two symbols is not exhaustive and the
  change may affect spot or another endpoint, so `normalize` will detect the
  unit rather than assume it — but nothing here reproduces it.
- **Archive sizes, measured 2026-03-02:** BTCUSDT and ETHUSDT 40 MB/day each,
  the other six configured symbols 1.8–6.9 MB/day. The ten-day eight-symbol
  pilot is 1.03 GB. Forty symbols over the same days is roughly 2 GB, not five
  times the pilot — the two majors dominate the total.
- **The HY ratio is not bounded by 1** (observed 1.016). A's return over one
  interval is counted against every overlapping B interval, so the numerator
  carries a multiplicity the realised variances do not. Confirmed against the
  literal double-sum definition. The v0.3 bootstrap must not assume a Fisher
  z-transform is defined.

## Open threads

**Decision: the estimator is built and proven before any real data is ingested**,
reversing the original v0.1/v0.2 milestone order. Ingestion is high-effort and
low-risk, and its failures are loud; the estimator is low-effort, high-risk, and
its failure mode is a plausible-looking wrong number. Only synthetic data has a
ground truth to check an estimator against, so real data cannot do this job.

In order:

1. ~~`TradeSeries` (`leadlag.types`)~~ — **done.** Validated in `__post_init__`,
   so an invalid series cannot exist; arrays frozen against in-place writes.
2. ~~`leadlag.synthetic`~~ — **done.** B a known delayed copy of A, independent
   Poisson arrivals, configurable trade-rate ratio. The measuring stick.
3. ~~`leadlag.estimators`~~ — **done.** Gridded baseline and Hayashi–Yoshida,
   verified against each other and against an O(n·m) oracle.
4. ~~exp001~~ — **done.** Gate 1 is green in CI and reproduced as an experiment
   with provenance and a figure.

5. ~~`leadlag.ingest`~~ — **done.** Verified, resumable, honest about gaps.
6. `leadlag.normalize` — Binance CSV to `TradeSeries`, including header
   detection, timestamp-unit detection and the tie-collapsing decided above.

Then ingestion, and the questions that follow it:

- Choose the lag grid: resolution and range. Too coarse hides the effect, too
  fine multiplies the hypothesis count that exp002 must correct for.
- Decide the universe and window length (affects the ~600-test count).

## Claims → evidence

| Claim in the report | Figure / table | Experiment |
| --- | --- | --- |
| The gridded estimator fabricates lead-lag from trade-rate asymmetry alone | `fig1_synthetic_recovery.pdf` (a, b) | exp001 |
| Hayashi–Yoshida recovers a known lag without a clock | `fig1_synthetic_recovery.pdf` (b) | exp001 |
| Lead–lag structure is real and survives multiple-testing correction | — | exp002 |
| Leadership rank is stable and tracks liquidity | — | exp003 |
| The effect is uneconomic at retail latency and fees | — | exp004 |
