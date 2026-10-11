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
| exp002 | What is the pairwise lag matrix across the universe, with bootstrap CIs and Benjamini–Hochberg correction over 28 tests? | **run, not trusted** — the tested statistic is uncalibrated | `outputs/exp002/` |
| exp003 | Is the directed lead–lag network stable across rolling windows, and does leadership rank track liquidity? | not started | — |
| exp004 | What is net Sharpe as a function of round-trip latency and fee level? | not started | — |
| exp005 | Under a known zero lag, how often do exp002's tests (curve asymmetry, peak lag) fire at nominal 5%? Decides which may enter BH. | **done** | both pass the pre-registered rule; asymmetry only narrowly. `outputs/exp005/` |
| exp006 | Are exp002's tests calibrated under a null with correlation *and* tick-signed bid-ask bounce? | **done** | both fail the pre-registered rule; the null itself is not yet validated. `outputs/exp006/` |

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
  change may affect spot or another endpoint, so `normalize` detects the unit
  rather than assuming it — but nothing here reproduces it. Header detection is
  verified against a real 2021 archive: 1,551,775 rows, no header row.
- **Collapsing simultaneous trades removes most of the deepest symbol.** On
  BTCUSDT 2026-03-02, 3,319,442 rows become 1,070,955 distinct milliseconds:
  67.7% of rows share a millisecond with another, in clusters of up to 263. On
  a quiet symbol (DOGEUSDT) it is 3.5%. This is the largest single
  transformation in the pipeline, and it puts a floor of roughly one
  millisecond on any lag that can be resolved — comfortably below the tens to
  hundreds of milliseconds under study, but the lag grid must not pretend to
  finer resolution than the data has.
- **No dependency was needed for parsing.** Stdlib `csv` runs at ~500k rows/s:
  8–9 s for a BTCUSDT day, about three minutes for the whole ten-day
  eight-symbol pilot, parsed once and cached. `pandas` and `pyarrow` both
  failed the justification test.
- **The canonical cache is 0.20–0.49x its source archive**, ~0.28x overall for
  the files held. High-volume symbols shrink most, because collapsing removes
  most of their rows. The 1.03 GB pilot implies roughly 260 MB of canonical
  data. Source archives are kept, so a bug in `normalize` costs a rerun and not
  a re-download.
- **Archive sizes, measured 2026-03-02:** BTCUSDT and ETHUSDT 40 MB/day each,
  the other six configured symbols 1.8–6.9 MB/day. The ten-day eight-symbol
  pilot is 1.03 GB. Forty symbols over the same days is roughly 2 GB, not five
  times the pilot — the two majors dominate the total.
- **On real data, nothing happens beyond ±50 ms.** On the coarse grid (50 ms
  steps to ±500 ms) 260 of 280 pair-days peak at ±0.5 ms, the rest at
  ±50.5 ms, never further out.
- **Inside ±50 ms there is structure.** On the fine grid (1 ms steps) the
  median |peak| is 4.5 ms, upper quartile 10.5 ms. Counting days, not yet a
  test: BTC leads XRP, LINK and AVAX on 10/10 days by a median 12.5–16.5 ms;
  ETH leads XRP, LINK and AVAX on 10/10 days by 10–13 ms; BTC leads ETH on
  8/10 by 11 ms; SOL leads LINK, DOGE and AVAX on 10/10 days, but by only
  1.5–5.5 ms. About half the pairs have no consistent sign. Unconfirmed until
  exp002 puts confidence intervals on it.
- **exp002 ran, and its significance test is not yet trustworthy.** Pooled
  over 10 days, 26 of 28 pairs pass BH at 5% on the curve-asymmetry statistic
  — most at the smallest q the 2000 draws allow. Three reasons to distrust it:
  (1) the asymmetries are tiny (0.001–0.02 on correlations of 0.08–0.27), so
  with 2,880 blocks almost anything is "significant"; (2) the pattern is
  per-symbol, not per-pair — ADA "leads" all seven others and AVAX trails all
  seven — which looks like a property of one symbol's price process rather
  than of information flow; (3) asymmetry and peak lag disagree: BTC→DOGE has
  a peak interval of [+9.5, +12.5] ms but asymmetry q = 0.75, and for ADA–SOL
  they point opposite ways. A zero-lag null on real trade clocks (one shared
  random walk observed at each symbol's real timestamps, one day) flagged 2 of
  12 runs at 95%; independent price noise on one side did not fake
  leadership. Not enough runs to calibrate anything.
- **Neither test is badly miscalibrated, but curve asymmetry fires twice as
  often as it should when the true lag is zero.** exp005, rule fixed before
  the run (both nulls; Wilson 95% interval of the false-positive rate at
  nominal 5% must not lie entirely above 5%). Shift null (real prices, B
  rotated 1–60 min, 140 tests): asymmetry 1.4% [0.4, 5.1], peak lag 0.7%
  [0.1, 3.9]. Latent null (real clocks, shared random walk, 84 tests):
  asymmetry **9.5% [4.9, 17.7]**, peak lag 2.4% [0.7, 8.3]. Both pass;
  asymmetry by 0.1 percentage points. Its p-values are anti-conservative
  under asynchronous sampling with strong correlation, which is exactly
  the real-data regime.
- **The calibration does not explain exp002's 26/28.** Under both nulls the
  largest |asymmetry| is 0.0016; 25 of exp002's 26 significant pairs exceed
  it, most by 3–10x. But neither null models microstructure *together with*
  correlation: the shift null destroys the correlation that a quirk like
  bid-ask bounce would need to show up in the cross-curve. So the per-symbol
  pattern (ADA "leads" all, AVAX trails all) is still unexplained, not
  ruled out as an artefact.
- **With correlation and bounce together, both tests fire far above 5%, and
  by the pre-registered rule neither may enter BH.** exp006, true lag zero:
  asymmetry 64/112 = 57.1% [47.9, 65.9], peak lag 32/112 = 28.6% [21.0, 37.5].
  No symbol was on one side of leadership in 6+ of 7 pairs, so criterion 2
  never triggered; criterion 1 alone excludes both. Read literally, exp002
  yields no finding. But a 57% false-positive rate from a half-tick bounce is
  large enough that the null may itself be wrong (tick-rule signs persist for
  long runs and are not tied to the latent path), so this result is a
  hypothesis about the estimator's failure, not yet a measurement of it.
  Unchecked: the same run with the bounce set to zero must reproduce
  exp005's latent rate (~9.5% asymmetry); if it does not, the fault is in the
  new code, not the estimator.
- **The peak lags, by contrast, look like the hypothesis.** Ordered by
  trading activity (ETH, BTC, SOL, XRP, DOGE, ADA, AVAX, LINK), ETH, BTC and
  SOL lead the five smaller symbols by 4.5–13.5 ms with bootstrap intervals
  clear of zero, and the five smaller symbols are mutually synchronous
  (±0.5 ms). Exceptions: ETH–BTC and ADA–BTC have intervals spanning both
  signs. This is an observation, not a result: switching the tested statistic
  to peak lag *because* it gives the expected picture would be choosing the
  test after seeing the answer.
- **A whole-millisecond lag grid biases Hayashi–Yoshida.** Timestamps are whole
  milliseconds, so at a whole-ms lag some A and B intervals share an endpoint
  exactly, and the half-open overlap rule drops their products. The curve is a
  sawtooth: correlation drops ~3–10% at every whole ms and is flat between.
  Ties are about twice as common at lag 0 as at 10 ms (BTC–AVAX: 9.1% vs 4.6%
  of B's trades), so the bias is not uniform and pushes peaks away from zero.
  Strong pairs moved ≤2.5 ms when ties were removed; weak pairs on flat days
  flipped sign (BTC–SOL 2026-03-26: −7 → +20.5 ms). **Decision: every lag
  sits half a millisecond off the clock**, ±(k·step + 0.5 ms), so a tie is
  impossible. Exactly 0 ms is no longer a candidate; the data could not
  resolve it anyway.
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
6. ~~`leadlag.normalize`~~ — **done.** Header and unit detection, collapsing,
   and a canonical cache.

7. ~~`leadlag.study`~~ — **done.** Hayashi–Yoshida over every pair in the
   8-symbol universe, one estimate per pair per full trading day, resumable
   canonical caching, gaps reported as `missing_data` rather than raised.

Decided: lag grid is 50ms steps, ±500ms (`configs/study.yaml`, same as
exp001's synthetic grid); universe is the 8 symbols already ingested, 28
pairs; one estimate per pair per full trading day, not a sub-day window.

The whole universe is ingested and cached: `make study` reports 280 `ok`, no
`missing_data`.

Decided: two lag grids, both half a millisecond off the clock
(`configs/study.yaml`): coarse ±0.5–500.5 ms in 50 ms steps, fine ±0.5–50.5 ms
in 1 ms steps. `make study` writes `outputs/study/results_coarse.csv` and
`results_fine.csv`.

exp002 is built (`make exp EXP=exp002`, ~3 min): HY on the fine grid, per-block
sums from `hayashi_yoshida_blocks`, 2000 multinomial block-bootstrap draws per
pair, BH at 5%. See "What we have learned" for why its output is not yet a
result.

Calibrated (exp005): both statistics pass the pre-registered rule; curve
asymmetry only narrowly, at twice its nominal false-positive rate under the
latent null.

**Decision rule for exp002, fixed before re-analysis.** The conclusion rests on
both statistics, not either alone. A pair is a *finding* only if (a) its curve
asymmetry passes BH at 5% over the 28 tests, (b) its peak-lag 95% bootstrap
interval excludes zero, and (c) the two agree on sign of leadership. A pair
passing exactly one is reported as *disputed*, with both values shown, never
dropped or promoted. Peak lag is not substituted for asymmetry because it gives
the expected picture; asymmetry's anti-conservatism (9.5% under the latent null)
is handled by requiring peak-lag agreement, not by changing its threshold. The
rule is not to be edited after seeing how many pairs it yields.

**exp006: a third null, rule fixed before it is run.** exp005's nulls each drop
something exp002's data has: the shift null keeps microstructure but destroys
correlation; the latent null keeps correlation but has no microstructure. exp006
is the latent null (real clocks, one shared random walk, true lag exactly zero)
plus a bid-ask bounce on every trade: ±h in log price, signed by the tick rule
(the sign of the last non-zero price change in that symbol's real series,
carried forward through zeros). The canonical cache has no `is_buyer_maker`, so
the tick rule is the only direction available; it keeps the real clustering of
trade signs, which a random sign would not. h is half the median absolute
non-zero log price change of that symbol-day, an approximation to half the
spread, not a measurement of it, and it is not to be tuned afterwards. 4
replicates, exp002's days, grid, blocks and bootstrap.

A statistic is excluded from exp002's BH step if, under this null, **either**
(1) the 95% Wilson interval of its false-positive rate at nominal 5% lies
entirely above 5%, **or** (2) in any replicate, some symbol is on the same
side of leadership in at least 6 of its 7 pairs among that statistic's
significant calls (p < 0.05). Criterion 2 is exp002's ADA/AVAX pattern: a
per-symbol artefact the null can reproduce is not evidence about information
flow. A statistic must also have passed exp005, so it is held to all three nulls.
If both statistics are excluded, that is reported and exp002 yields no finding.

Next: validate exp006's null before believing it. Run it with
`half_spread_fraction: 0` as a diagnostic (not a re-tune of the rule): it must
reproduce exp005's latent rates. If it does, the finding is that
Hayashi-Yoshida's asymmetry and peak-lag tests are unusable on real
microstructure, and the next question is a bounce-robust estimator (e.g.
pre-averaging or a minimum-interval filter) rather than exp002 conclusions.

## Claims → evidence

| Claim in the report | Figure / table | Experiment |
| --- | --- | --- |
| The gridded estimator fabricates lead-lag from trade-rate asymmetry alone | `fig1_synthetic_recovery.pdf` (a, b) | exp001 |
| Hayashi–Yoshida recovers a known lag without a clock | `fig1_synthetic_recovery.pdf` (b) | exp001 |
| Lead–lag structure is real and survives multiple-testing correction | — | exp002 |
| Leadership rank is stable and tracks liquidity | — | exp003 |
| The effect is uneconomic at retail latency and fees | — | exp004 |
