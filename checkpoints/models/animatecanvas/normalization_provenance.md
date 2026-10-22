# HYMotion M2M Strict-198 Normalization Stats

These stats are derived directly from the official HY-Motion 1.0-Lite 201-dim
normalization stats in:

```text
checkpoints/HY-Motion-1.0/stats/{Mean.npy,Std.npy}
```

The strict 198-dim representation is:

```python
idx = list(range(135)) + list(range(138, 201))
motion198 = motion201[..., idx]
```

Therefore:

```python
Mean198 = Mean201[idx]
Std198 = Std201[idx]
```

The dropped `motion201[..., 135:138]` block is the pelvis RIC triplet.  For the
official HYMotion 201-dim representation this triplet has zero data, zero mean,
and zero std:

```text
Mean201[135:138] = [0, 0, 0]
Std201[135:138]  = [0, 0, 0]
```

This file replaces the older M2M-native Scheme-D stats, where non-pelvis joint
Y positions used absolute world height.  Those old stats were incompatible with
HY-Motion 1.0-Lite warm-start because the checkpoint adapter maps 201-dim IO
weights by dropping only the pelvis RIC triplet.
