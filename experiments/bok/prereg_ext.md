# Pre-registration: extension runs for the ICEIC 2027 paper (written 2026-10-16, before any of these runs)

Deadline moved to 2026-10-30. These runs answer the weaknesses found in the review of 2026-10-16
(reviews/iceic2027_improvement_plan_2026-10-30.md). Reporting rules are fixed here, before results exist.
Freeze: no run started after 2026-10-24 enters the submission.

## B4. Table II on all 500 episodes; undisturbed cell of the scripted shift
- Teacher res_v4.pt.it039 (--gate jump, static schedule) on initial states 0-24, noise salts 0-4 (the held-out
  draws 25-49 with salts 0-4 already exist). Merged per salt into five all-500 draws. Undisturbed: inits 0-24, salt 0,
  merged with the existing held-out undisturbed run.
- Scripted shift (scripted:0.5,2, default cumulative gate, as in every scripted number of the paper) on inits 0-24,
  salts 1-2, merged per salt with the existing held-out draws (salt 0 already covers all 500): three all-500 draws.
  Undisturbed: all 500, same cumulative gate. This replaces the jump-gated 95.2 now in Table II, whatever its value.

## B1. Fixed-rule shift over a longer window (is the learned residual needed?)
- Candidates vision:(alpha,k): (0.5,2) [current], (0.5,4), (1.0,4), (0.5,6), (0.75,6), (1.0,6), (1.5,6).
- Selection uses training initial states 0-24 only. Stage 1: one draw each (salt 0; the current setting uses its
  existing rows on 0-24). Stage 2: the two best by stage-1 success get a second draw (salt 1; current: existing ns1).
  Winner = higher two-draw mean success; ties -> smaller k, then smaller alpha.
- If the winner is not (0.5,2): evaluate it on all 500 episodes with salts 0-2, undisturbed (salt 0, all 500) and at
  8-12 cm (salts 0-1). The paper then compares the student with the selected rule and states the selection
  (on training initial states), whatever the result. If the winner is (0.5,2), the paper says that longer windows
  were tried and did not help on training states.

## B5. More draws for single-draw arms
- 8-12 cm: fixed rule (0.5,2) salts 1-2, student without d-hat salt 1, base salt 2.
- Late trigger (time, steps 100-200): teacher salt 1, student salt 1, base salt 1.
- Report pooled over all draws; if an interval then includes zero, the sentence is weakened or removed.

## B2. Training-seed repeats of the student (DAgger)
- student_distill.py --seed 1 and 2 (seed 0 = the released student_v2, unchanged), same teacher, detector,
  regressor and settings; also the no-d-hat student with seeds 1 and 2.
- Evaluation: student seeds all 500 with salts 0-1, undisturbed salt 0, 8-12 cm salt 0; no-d-hat seeds all 500 salt 0
  and 8-12 cm salt 0.
- Reporting: the released module (seed 0) stays the deployed row; the paper adds the mean and the range over the
  three seeds. A better seed is never swapped in.

## B6. openpi's default execution setting (10,5) under the disturbance
- Base policy at (n,h)=(10,5), all 500 disturbed episodes, salts 0-1; undisturbed (10,5) already has three runs
  (91.6-92.2%). Report the disturbed success and its drop next to (2,10) whatever the result.
