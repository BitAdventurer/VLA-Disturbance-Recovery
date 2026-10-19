# Pre-registration: one-time fresh-initial-state test (ICEIC 2027 paper), 2026-10-19

Written and hashed before any rollout on these states. Nothing below is changed after the hash; any deviation is
reported as a deviation.

## Purpose
The paper's "held-out" initial states 25-49 were held out from training but used to choose gates, alpha and model
versions. This test evaluates the pipeline as submitted, once, on initial states that no rollout of this project has
used.

## Population
- LIBERO-10, the full `.init` file of each task, states 50-99: 10 tasks x 50 = 500 episodes per arm-draw.
- Audit of 2026-10-18 (read-only, independently re-checked on 2026-10-19):
  - `.pruned_init` equals `.init[0:50]` bit for bit for all 10 tasks.
  - No state in 50-99 duplicates one in 0-49.
  - No rollout, training run, selection or stored result of this project used states 50-99. The 375 result files have a maximum init of 49. Only read-only placement audits read their object positions (2026-10-18, and the extension track's support-set analysis), and they ran no rollouts.
  - The scene-level nearest-neighbour distance from 50-99 to 0-49 (median 16.4 mm) equals the distance within 0-49 (16.3 mm).
- These are new initial states of the same tasks from the same placement distribution, not new tasks.
- Settle outliers (a free object moving more than 5 mm in xy during the 10 settle steps: task 1 #87, task 6 #52, task 7 #76) stay in every endpoint. One sensitivity line gives P1 without them.

## Locked configuration
- Policy: the released pi0.5 LIBERO checkpoint (openpi pi05_libero, PyTorch), served by `scripts/start_servers.sh` (`serve_policy_dynamic.py`), with (n, h) = (2, 10).
- Disturbance: the `residual_rl.py` defaults used in the paper. An object is teleported 4-6 cm when the open gripper first comes within 12 cm. The fallback time is drawn from the window 40-160 after 10 settle steps.
- Detector: `bok/detector_v1_t24.pt`, step_tau 1.628, react_setting 2,10.
- Regressor: `bok/dispreg_v2_t24.pt`.
- Student: `ckpt/student_v2.pt` (released), gate jump.
- Fixed rule: vision:0.5,2. The rule selected on training states (alpha 0.5, k 4) is not re-tested here; that comparison stays on states 0-49.
- Episode keys: (0, eval_seed 0, task, init 50..99). These are new keys, so every displacement draw is new.
- Draws vary only the policy's sampling noise. The displacement is the same in every arm and draw of an episode.
- Checkpoints equal the public release byte for byte (sha256 below).

## Harness
- Every arm runs through `residual_rl.py --eval` with `--init_file full --eval_init_lo 50 --eval_init_hi 100`. The base uses `--eval zero`.
- The option is off by default. With it off, outputs are byte-identical to the previous code (mock tests t1-t3). The silent modulo wrap is replaced by an explicit range error.
- The run uses the tested patched copy in `~/disk2/vla-env/run_fresh50`; the live `experiments/` tree is not modified. One policy server and 10 rollout workers.
- The script is `run_fresh50.sh`. Outputs go to `experiments/results_test/`, which the paper's held-out analysis never reads.
- Analysis: the block `fig_paper_fresh50_block.py`, inserted verbatim into `fig_paper.py`. It uses the `paired()` defined there, lists every input file explicitly and writes `extra.fresh50` to `res_numbers.json`. Paper numbers come only from it.

## Arms and draws (file names in results_test/)

| Role | Arm | Flags | Files |
|---|---|---|---|
| reference | base | `--eval zero` | fresh_base_s0, fresh_base_ns1, fresh_base_ns2, fresh_base_ns3, fresh_base_ns4 |
| primary | residual student | `--eval student:ckpt/student_v2.pt` + detector + `--gate jump` | fresh_student_s0, fresh_student_ns1, fresh_student_ns2 |
| primary | fixed-rule shift | `--eval vision:0.5,2` + detector | fresh_fixed_s0, fresh_fixed_ns1, fresh_fixed_ns2 |
| secondary | detector -> replan | `--eval zero` + detector | fresh_det_s0, fresh_det_ns1 |
| secondary | undisturbed base / student | `--eval_disturb none` | fresh_none_base_s0 / fresh_none_student_s0 |

## Endpoints
All endpoints are on the 500 fresh episodes. Differences are paired, using the per-episode mean over draws. Intervals
are task-stratified bootstrap 95% intervals (`paired()`, 6,000 resamples, seeded by the comparison). An episode enters a
comparison if it is present in every reference draw and in at least one arm draw. No substitute salt is ever run.
- **P1 (the single confirmatory test, two-sided 95%):** student minus base.
- **S1:** student minus fixed-rule shift.
- **S2:** fixed-rule shift minus base.
- **S3 (descriptive):** detector replan minus base. An interval that includes 0 is reported as inconclusive.
- **S4:** undisturbed student minus undisturbed base at the same salt, with the episodes that flipped and the fraction in which the student acted.
- **S5 (exploratory):** P1 and S1 split by trigger. An episode counts as a time fallback if, in a majority of the five base draws, applied_t equals the planned step recomputed from the key (`run_libero.disturbance_plan`); otherwise it is approach-triggered.
- **Sensitivity:** P1 without the three settle outliers.
- **Expected half-widths:** about 3.0 (P1), 2.5 (S1), 2.8 (S2) and 2.6 (S3) points. These are recomputed on states 0-49 with the same draw counts.
- S1-S5 are unadjusted secondary results.

## Integrity
- Every arm-draw must have 500 rows covering the 500 distinct (task, init) keys.
- An incomplete arm-draw gets at most two attempts over all invocations of the script, and earlier outputs and logs are kept.
- If the runner itself is interrupted (server or machine failure), it is restarted. Complete arm-draws are kept, and every attempt log is kept and reported. A still-incomplete arm-draw is reported as such, and the analysis states the n it used.
- No arm, draw, setting or endpoint is added, dropped or changed after any result is seen.
- States 50-99 are not used again for any selection or tuning.

## Reporting
- P1 and S1 go into the paper whatever their values. S2-S5 and the sensitivity line are reported in the results or the code release whatever their values.
- **P1:** if its interval excludes 0, the abstract states that the gain holds on fresh initial states. Otherwise it states that the gain was not confirmed, and the 25-49 numbers are qualified as affected by selection.
- **S1:** the same rule applies to the sentence that the student adds over the fixed-rule shift.
- **S4:** the same rule applies to "no measurable change in undisturbed success".
- **Earlier results:** those on states 25-49 stay as reported, described as held out from training and used for model selection. The difference between fresh and 25-49 results is reported descriptively, without a test.

## Hashes (sha256)
Filled in by `prereg_fresh50.sha256`, written at the same time:
- this file
- `run_fresh50.sh`, `fig_paper_fresh50_block.py`, `patch_fresh50.py`
- the patched `residual_rl.py` and `run_libero.py`
- `ckpt/student_v2.pt`, `bok/detector_v1_t24.pt`, `bok/dispreg_v2_t24.pt`
