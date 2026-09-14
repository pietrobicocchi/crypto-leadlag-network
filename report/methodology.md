# Methodology

The argument this project makes, and what it does not establish. Figures
referenced here live in `report/figures/` and are promoted from `outputs/` by
`make figures`. Canonical implementations live in `src/leadlag/` — point at
them rather than restating them.

## Thesis

Lead–lag structure between crypto instruments is real and measurable at the
sub-second scale, and it is not economically exploitable at retail latency and
fee levels.

## Argument

1. The standard gridded approach manufactures a lag from trade-rate asymmetry
   alone, so any leadership it reports is an artefact. → evidence: (exp001)
2. The Hayashi–Yoshida estimator recovers a known lag on asynchronous data with
   no clock and no interpolation. → evidence: (exp001)
3. Applied to the universe, it finds a lag structure that survives bootstrap
   confidence intervals and Benjamini–Hochberg correction. → evidence: (exp002)
4. That structure is stable across rolling windows, and leadership rank tracks
   liquidity. → evidence: (exp003)
5. Under an event-time simulation with realistic latency and fees, the net
   Sharpe is indistinguishable from zero. → evidence: (exp004)

## Results

<What the figures show, in the order the reader meets them. Not yet written.>

## Methods

<Estimator definitions, the lag grid, the bootstrap scheme and the
multiple-testing correction. Point at `src/leadlag/`. Not yet written.>

## Limitations

Written before the results, so that they are not written to excuse them.

- **Trades, not quotes.** `aggTrades` records executions. Price discovery
  happens in the order book, so a lag measured on trades is a lower bound on
  how fast information actually moves.
- **One venue.** Everything here is Binance. Cross-venue lead–lag is a
  different and probably larger effect, and is not measured.
- **Exchange timestamps.** Times are as stamped by the matching engine, not as
  observed by any participant. Clock behaviour is assumed, not verified.
- **The simulator is not a backtest.** It assumes fills at quoted prices with a
  fixed latency and fee. It models no queue position, no market impact, and no
  adverse selection — all of which make a real result worse, never better.
- **Replication, not discovery.** The effect and the estimator are both known.
  The contribution here is engineering and honest measurement.

## Cut

<Findings that are true but not part of this report. Keep them here rather than
deleting them.>
