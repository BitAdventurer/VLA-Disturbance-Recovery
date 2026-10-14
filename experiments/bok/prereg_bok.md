# Pre-registration: best-of-K chunk selection, signal phase (written 2026-10-02 12:10 KST, before any data)

Setting: frozen pi0.5 (PyTorch, RTX 5090), LIBERO-10, (n,h) = (2,10), approach-triggered object displacement 4-6 cm,
episode keys (0, 0, task, init 0..49). Baseline = mean of 4 noise draws of (2,10): 64.9%
(s0 67.8, ns1 65.0, ns2 63.4, ns3 63.2).

## Arms
All in seq mode (K ordinary requests per query). W = 7 queries from the first query after the displacement
(P90 of the number of queries until the first gripper close).
- BR0: branch runs on bok/strat.json (284 episodes: all 182 mixed, 50% of the 88 always-fail, 25% of the 230
  always-success; Horvitz-Thompson weights 1, 2, 4), fresh env per episode, forced candidate k in 0..3 at the first
  query after the displacement, continuation salt c in {0, 1}. BR2: the same at the third query after it.
- SEL: K=8 inside the window with the selectors priv_aim (privileged: true object position; a diagnostic ceiling,
  never a deployable method), random, medoid, continuity; K=16 priv_aim; antimedoid, discontinuity.

## Statistics
Primary: split oracle G(R) per episode = 1/2 * sum over (c, c2) in {(0,1), (1,0)} of
[ mean over argmax_k Y(k,c) of Y(k,c2) - mean_k Y(k,c2) ], Hajek-weighted over strata, bootstrap within strata
(10k resamples). Unbiased for the gain of an oracle that picks the best of 4 candidates using one continuation; a
lower bound on a perfect oracle.
Secondary: Delta(arm) = success(arm) - per-episode mean of the 4 baseline draws; task-stratified paired bootstrap.

## Gates
- GO (critic track): G(0) >= +4.0 points with 95% CI lower bound > +1, or priv_aim@8 Delta >= +8 (single draw)
  and >= +6 after two confirmation salts.
- LIMITED: G(0) in [+2, +4) or priv_aim@8 Delta in [+4, +8): critic work capped at 1.5 days.
- HEURISTIC: a non-privileged selector with Delta >= +5, confirmed with two more salts (CI lower bound > 0).
- NO-GO: G(0) 95% upper bound < +2.5 and priv_aim@8 Delta < +4: stop chunk selection; write up the negative result.

Replay check: prefix_ok = executed-chunk hashes before the branch query are equal across the 8 runs of an episode.
Its rate is reported (a rare simulator/renderer nondeterminism exists); G is computed on all episodes and on the
prefix_ok episodes.
