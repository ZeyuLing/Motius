# MotionCanvas position IK

This is the engineering position-projection implementation, not the temporal
SQP solver described in the paper. No paper metrics are populated by this change.

## Changes

- Two-bone targets no longer shrink to 99% reach. An analytic limb solve uses
  the current bend plane when well defined. It also handles full extension.
- Coupled cues on one frame are solved simultaneously with an analytic
  tangent-space Jacobian and damped least squares. No Adam pose penalty trades
  away accuracy. A line search and best-maximum-residual return avoid returning
  a worse iterate merely because it was last.
- Frames with matching cue layouts and batch examples are evaluated together.
- `PositionConstraint(..., axes=(True, False, True))` controls XZ only.
- `solve(..., fixed_mask=...)` preserves input translation coordinates and
  whole six-dimensional rotation blocks. Partial rotation blocks are rejected.
  For global-rotation inputs, a locked global rotation conservatively locks
  its local ancestors as well. This is safe but may reduce reachability.
- Unconstrained root translation is not used to cheat position accuracy. It
  stays fixed unless a root-position cue explicitly enables those coordinates.
- Inputs can be motion135 or motion198, and can include a batch. Extra channels
  are preserved exactly. Only FK from translation/rotations determines errors.
- The pipeline uses sequence-relative frame indices, rejects padding cues,
  protects known rotation/translation channels, and checks errors after final
  replacement and normalization round trips.

## Units and convergence

The API uses meters. The default tolerance is `1e-5` meters (0.01 mm), leaving
headroom for an external 0.1 mm success threshold. This is a target, not a
guarantee. All-cue success must be evaluated on the returned motion. Unreachable,
inconsistent, locked or numerically stalled constraints retain a nonzero error.

The solver returns `(motion, max_error_m)`. Pipeline outputs also contain
`position_constraint_max_error_m` and `position_constraint_satisfied` (the latter
uses the solver's tolerance). There is no claim of guaranteed zero error,
temporal smoothness, joint-limit satisfaction or collision avoidance.

Unconstrained ancestor rotations may change during DLS fallback. A successful
position projection is not proof that motion quality is unchanged. Check Foot,
Jitter and renders on real outputs before enabling the path in a production run.

## Validation

Run in an approved environment with the Motius core dependencies:

```console
python -m pytest tests/test_position_constraint_ik.py tests/test_motioncanvas.py -q
python tools/benchmark_position_ik.py --device cpu --frames 120 --repeats 5
python tools/benchmark_position_ik.py --device cuda --frames 300 --repeats 20
```

The benchmark is synthetic, not P1--P4 or Table 18 evaluation. It fixes seed 42,
warms up, synchronizes CUDA, and includes conversion/solver/final FK checking in
the IK time. It excludes generation and reports **cue-level** P95, explicitly
not a percentile across per-sequence maximum errors. Repeated timing uses the
same motion; it is not a success-rate estimate over independent test motions.

Status on 2026-09-16: source compilation and whitespace validation passed.
Numerical tests and timing are pending: the current Windows host's Smart App
Control blocked both the project Python runtime and PyTorch's native library.
No security policy was changed and no runtime results are claimed.
