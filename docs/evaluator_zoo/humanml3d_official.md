<h1 align="center">HumanML3D Official Evaluator Card</h1>

<p align="center">
  <strong>Official HumanML3D / T2M leaderboard metric view.</strong>
</p>

HumanML3D Official is the native metric view for methods that generate
HumanML3D-263 features. Motius uses the selected-caption HumanML3D test
protocol for public model-card reporting.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Evaluator | HumanML3D Official |
| Motion representation | HumanML3D-263 |
| Caption protocol | Selected caption for HumanML3D test split |
| Metrics | R@1, R@2, R@3, FID, MM-Dist, Diversity |
| Checkpoint/assets | Official HumanML3D evaluator assets |

## Reporting Rule

Every T2M model card should include this row. If a method is not evaluated in
native HumanML3D-263 space yet, the row should be marked `Pending`, not replaced
by a different evaluator.

## Notes

For FID and MM-Dist, lower is better. For R-Precision and Diversity, higher is
usually better, but Diversity should be compared together with the ground-truth
row and the target dataset protocol.
