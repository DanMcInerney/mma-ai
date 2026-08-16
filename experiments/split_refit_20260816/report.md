# Chronological split and full-refit evidence report

## Decision

Retain the immutable weighted-v8 rollback as the production fallback. Preserve the fresh full-data refit as a loadable, non-validated deployment candidate; do not promote it from FULL evidence.

## Historical evaluation evidence

All three results are historical and selection-exposed to different degrees. Their denominators represent different protocols and are not pooled or treated as directly equivalent. This campaign does not establish untouched, external, or prospective performance.

| protocol | correct / rows | accuracy | positive log loss |
| --- | ---: | ---: | ---: |
| accepted direct tuning | 309 / 460 | 0.671739 | 0.613185 |
| nested whole-event 2022–2025 | 726 / 1,108 | 0.655235 | 0.619595 |
| one-shot chronological test | 202 / 307 | 0.657980 | 0.626169 |

The one-shot event-block 95% interval is [0.611650, 0.707237] for accuracy and [0.590978, 0.662629] for positive log loss.

## Full-data refit boundary

The saved predictor loads with 22 nodes and selects `WeightedEnsemble_L2_FULL`. Nine FULL base nodes are fresh 3,267-row fits. RealMLP_r9_FULL is an Original clone fitted on 2,807 rows. The FULL ensemble wrapper retains 460-row metadata but has effective 3,267-row lineage through `Mitra_FULL` and `XGBoost_FULL` (weights 0.96/0.04).

No validation metric is claimed for the full-data refit. The only production process exited 1 after training, `refit_full`, and permutation importance completed; the order-assertion failure is preserved, read-only recovery mutated no model, and retry count is zero.

## Branches and rollback

- rollback: `codex/weighted-v8-67-baseline` at `545441975b86caf0abb6136e099e44e6b93caf22`
- evaluation: `codex/exp-80-10-10-v8-20260816` at `7217012abcee3c22937dd378c0a904033564018d`
- full refit: `codex/exp-full-refit-v8-20260816` at `70559ac40300c62067f23b335050dda3e4931ce6`

Verify the existing rollback worktree without touching the original dirty checkout:

```powershell
git -C C:/Users/danhm/mma-ai/mma-ai rev-parse codex/weighted-v8-67-baseline
git -C C:/Users/danhm/mma-ai/worktrees/weighted-v8-67-baseline rev-parse HEAD
git -C C:/Users/danhm/mma-ai/worktrees/weighted-v8-67-baseline status --short
```

The first two commands must both print the fixed rollback revision above; the final command must print nothing.

## Replacement predicate

Keep the rollback if any retain predicate holds. Replace it only when every preregistered replace predicate passes on future outcome-unknown whole-event evidence, with no post-result adaptation and full identity replay.
