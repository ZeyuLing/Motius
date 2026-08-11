"""Task-agnostic sampler for arbitrary motion evidence.

The sampler operates on the semantic grid ``time x atom``.  An atom is one
translation axis, one complete 6D joint rotation, or one joint-position axis.
Every condition clause first samples an explicit temporal density and topology,
then samples its coordinate content. Most masks are Boolean Rank-K layouts:
independently sampled temporal and coordinate subsets are combined with outer
products, then merged by OR. A random-field branch provides highly fragmented
layouts useful for repair-like completion while obeying the same temporal
density contract.

The structured components redistribute finite-sample probability at the
primitive level.  They do not enumerate trajectory, keyframe, completion,
body-part, or repair tasks.

Mask convention: ``1=generate`` and ``0=known evidence``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

MOTION_DIM = 198
NUM_JOINTS = 22
NUM_POSITION_JOINTS = 21
ROT_START = 3
POS_START = 135

SMPL22_PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int64,
)

SMPL22_MIRROR = np.asarray(
    [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15,
     17, 16, 19, 18, 21, 20],
    dtype=np.int64,
)


@dataclass(frozen=True)
class MotionAtom:
    """Smallest independently conditionable semantic motion unit."""

    kind: str
    joint: int
    axis: int
    dims: Tuple[int, ...]


def _build_atoms() -> Tuple[MotionAtom, ...]:
    atoms: List[MotionAtom] = []
    for axis in range(3):
        atoms.append(MotionAtom("translation", -1, axis, (axis,)))
    for joint in range(NUM_JOINTS):
        start = ROT_START + 6 * joint
        atoms.append(MotionAtom("rotation", joint, -1, tuple(range(start, start + 6))))
    for joint in range(1, NUM_JOINTS):
        for axis in range(3):
            dim = POS_START + 3 * (joint - 1) + axis
            atoms.append(MotionAtom("position", joint, axis, (dim,)))
    return tuple(atoms)


ATOMS = _build_atoms()
NUM_ATOMS = len(ATOMS)  # 3 + 22 + 21 * 3 = 88

TRANSLATION_ATOMS = np.arange(0, 3, dtype=np.int64)
ROTATION_ATOMS = np.arange(3, 3 + NUM_JOINTS, dtype=np.int64)
POSITION_ATOMS = np.arange(3 + NUM_JOINTS, NUM_ATOMS, dtype=np.int64)

# These are primitive-level priors, not task probabilities.  The random-field
# branch below gives every mask positive probability; this branch gives useful
# finite-sample mass to repeated time/coordinate structures.
RANDOM_FIELD_PROB = 0.30
SINGLE_CLAUSE_PROB = 0.45
TIME_ALL_PROB = 0.45
REUSE_TIME_ATOM_PROB = 0.30
RANK_WEIGHTS = np.asarray(
    (0.28, 0.30, 0.20, 0.12, 0.05, 0.025, 0.015, 0.01),
    dtype=np.float64,
)

# A root trajectory supplies only 2-3 observed atoms but must determine the
# dynamics of the entire body. Give pure translation clauses enough finite-
# sample mass to learn that one-to-many coupling without adding a task-specific
# trajectory branch. All modalities remain supported by the random field and
# Boolean Rank-K branches.
COORDINATE_TYPE_WEIGHTS = np.asarray(
    (0.10, 0.38, 0.14, 0.20, 0.18),
    dtype=np.float64,
)

# Coordinate cardinality is independent of temporal density. In particular,
# changing a training config's density_bin_weights must only change how many
# frames are observed, not how many joints or axes are selected on those frames.
COORDINATE_DENSITY_BIN_WEIGHTS = (0.30, 0.25, 0.18, 0.14, 0.13)

AXIS_SUBSETS = (
    (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2),
)
AXIS_SUBSET_WEIGHTS = np.asarray(
    (0.08, 0.06, 0.08, 0.13, 0.28, 0.13, 0.24),
    dtype=np.float64,
)

# Translation gets its own primitive prior so XZ/XYZ trajectories become
# frequent without weakening arbitrary per-joint position-axis coverage.
TRANSLATION_AXIS_SUBSET_WEIGHTS = np.asarray(
    (0.06, 0.04, 0.06, 0.10, 0.34, 0.10, 0.30),
    dtype=np.float64,
)


def _graph_distances() -> np.ndarray:
    adjacency = [[] for _ in range(NUM_JOINTS)]
    for joint, parent in enumerate(SMPL22_PARENTS):
        if parent >= 0:
            adjacency[joint].append(int(parent))
            adjacency[int(parent)].append(joint)

    distances = np.full((NUM_JOINTS, NUM_JOINTS), 999.0, dtype=np.float32)
    for source in range(NUM_JOINTS):
        distances[source, source] = 0.0
        frontier = [source]
        while frontier:
            current = frontier.pop(0)
            for neighbor in adjacency[current]:
                candidate = distances[source, current] + 1.0
                if candidate < distances[source, neighbor]:
                    distances[source, neighbor] = candidate
                    frontier.append(neighbor)
    return distances


GRAPH_DISTANCES = _graph_distances()


def _standardize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    return (values - values.mean()) / max(float(values.std()), 1e-6)


def _ar1_field(length: int, rng: np.random.RandomState, scale: float) -> np.ndarray:
    innovation = rng.normal(size=length).astype(np.float32)
    phi = float(np.exp(-1.0 / max(scale, 1e-3)))
    sigma = float(np.sqrt(max(1.0 - phi * phi, 1e-6)))
    result = np.empty(length, dtype=np.float32)
    result[0] = innovation[0]
    for index in range(1, length):
        result[index] = phi * result[index - 1] + sigma * innovation[index]
    return _standardize(result)


def _atom_field(rng: np.random.RandomState) -> np.ndarray:
    """Correlated random field over joints, modalities, and position axes."""
    score = rng.normal(size=NUM_ATOMS).astype(np.float32)
    kind_bias = {
        kind: float(rng.normal())
        for kind in ("translation", "rotation", "position")
    }
    axis_bias = rng.normal(size=3).astype(np.float32)

    center = int(rng.randint(0, NUM_JOINTS))
    radius = float(np.exp(rng.uniform(np.log(0.5), np.log(8.0))))
    amplitude = float(rng.normal(scale=1.5))
    graph_bias = amplitude * np.exp(-GRAPH_DISTANCES[center] / radius)

    for index, atom in enumerate(ATOMS):
        score[index] += kind_bias[atom.kind]
        if atom.axis >= 0:
            score[index] += axis_bias[atom.axis]
        if atom.joint >= 0:
            score[index] += graph_bias[atom.joint]
    return _standardize(score)


def _sample_known_count(
    size: int,
    rng: np.random.RandomState,
    density_bin_weights: Tuple[float, ...],
) -> int:
    """Sample every non-degenerate cardinality with positive probability."""
    if len(density_bin_weights) != 5:
        raise ValueError("density_bin_weights must contain five entries")
    weights = np.asarray(density_bin_weights, dtype=np.float64)
    if np.any(weights <= 0.0):
        raise ValueError("every density bin must have positive probability")
    weights /= weights.sum()

    # The first interval is open at zero; all others are left-open/right-closed.
    boundaries = (0.0, 0.05, 0.20, 0.50, 0.80, 1.0)
    density_bin = int(rng.choice(5, p=weights))
    low = max(1, int(np.floor(boundaries[density_bin] * size)) + 1)
    high = min(size - 1, int(np.floor(boundaries[density_bin + 1] * size)))
    if high < low:
        return int(np.clip(low, 1, size - 1))
    if density_bin == 0 and high > low:
        # Log-uniform cardinality prevents the ultra-sparse range from being
        # dominated by its largest counts while preserving support for every
        # admissible count in (0, 5%].
        count = int(np.floor(np.exp(rng.uniform(np.log(low), np.log(high + 1)))))
        return int(np.clip(count, low, high))
    return int(rng.randint(low, high + 1))


def _sample_score_grid(
    length: int,
    rng: np.random.RandomState,
    *,
    max_rank: int,
    iid_floor: float,
) -> np.ndarray:
    """Sample a full-support random field over ``time x semantic atom``."""
    if not 0.0 < iid_floor < 1.0:
        raise ValueError("iid_floor must be in (0, 1)")

    iid = _standardize(rng.normal(size=(length, NUM_ATOMS)))

    temporal_scale = float(
        np.exp(rng.uniform(np.log(0.35), np.log(max(length, 1))))
    )
    temporal = np.broadcast_to(
        _ar1_field(length, rng, temporal_scale)[:, None],
        (length, NUM_ATOMS),
    )
    atom = np.broadcast_to(_atom_field(rng)[None, :], (length, NUM_ATOMS))

    rank = int(rng.randint(1, max_rank + 1))
    low_rank = np.zeros((length, NUM_ATOMS), dtype=np.float32)
    for _ in range(rank):
        scale = float(np.exp(rng.uniform(np.log(0.35), np.log(max(length, 1)))))
        time_factor = _ar1_field(length, rng, scale)
        atom_factor = _atom_field(rng)
        low_rank += np.outer(time_factor, atom_factor).astype(np.float32)
    low_rank = _standardize(low_rank)

    # Alpha < 1 puts useful mass near each simplex corner while retaining all
    # mixtures. Independently varying the IID share spans highly fragmented,
    # mixed, and strongly correlated masks without naming any task family.
    mixture = rng.dirichlet(np.full(3, 0.35, dtype=np.float64))
    structured = (
        mixture[0] * temporal
        + mixture[1] * atom
        + mixture[2] * low_rank
    )
    correlation_regime = int(rng.choice(3, p=(0.30, 0.40, 0.30)))
    if correlation_regime == 0:
        iid_weight = float(rng.uniform(0.65, 0.95))
    elif correlation_regime == 1:
        iid_weight = float(rng.uniform(0.15, 0.50))
    else:
        iid_weight = float(
            np.exp(rng.uniform(np.log(iid_floor), np.log(0.08)))
        )
    score = iid_weight * iid + (1.0 - iid_weight) * _standardize(structured)
    return _standardize(score)


def _sample_rank(
    max_rank: int,
    rng: np.random.RandomState,
    *,
    min_rank: int = 1,
) -> int:
    weights = RANK_WEIGHTS[:max_rank].copy()
    if max_rank > len(weights):
        tail = 0.5 ** np.arange(1, max_rank - len(weights) + 1)
        weights = np.concatenate([weights, 0.0025 * tail])
    min_rank = int(np.clip(min_rank, 1, max_rank))
    weights[:min_rank - 1] = 0.0
    weights /= weights.sum()
    return int(rng.choice(np.arange(1, max_rank + 1), p=weights))


def _sample_axis_subset(
    rng: np.random.RandomState,
    weights: np.ndarray = AXIS_SUBSET_WEIGHTS,
) -> Tuple[int, ...]:
    weights = weights / weights.sum()
    return AXIS_SUBSETS[int(rng.choice(len(AXIS_SUBSETS), p=weights))]


def _sample_temporal_atom(
    length: int,
    rng: np.random.RandomState,
    density_bin_weights: Tuple[float, ...],
) -> np.ndarray:
    """Sample a generic frame subset, from persistent to fragmented."""
    if length == 1:
        return np.ones(1, dtype=np.uint8)
    if rng.random() < TIME_ALL_PROB:
        return np.ones(length, dtype=np.uint8)

    # Cardinality is drawn exactly once. Every topology below must preserve it,
    # so the selected density bin has an unambiguous meaning.
    count = _sample_known_count(length, rng, density_bin_weights)
    mode = int(rng.choice(6, p=(0.09, 0.50, 0.09, 0.13, 0.11, 0.08)))
    selected = np.zeros(length, dtype=np.uint8)

    if mode == 0:
        # Boundary intervals give sequence endpoints the same primitive-level
        # status as interior intervals without fixing a benchmark ratio.
        if rng.random() < 0.5:
            selected[:count] = 1
        else:
            selected[length - count:] = 1
    elif mode == 1:
        # Split the already sampled total cardinality across both boundaries.
        # Sampling another count here would silently double temporal density.
        left_count = count // 2
        right_count = count - left_count
        selected[:left_count] = 1
        selected[length - right_count:] = 1
    elif mode == 2:
        start = int(rng.randint(0, length - count + 1))
        selected[start:start + count] = 1
    elif mode == 3:
        indices = rng.choice(length, size=count, replace=False)
        selected[indices] = 1
    elif mode == 4:
        scale = float(np.exp(rng.uniform(np.log(0.35), np.log(max(length, 1)))))
        selected = _topk_mask(_ar1_field(length, rng, scale), count)
    else:
        # Multiple random attraction centres produce disconnected spans with
        # arbitrary locations and widths, without enumerating task layouts.
        frame_index = np.arange(length, dtype=np.float32)
        num_centres = int(rng.randint(2, min(5, count + 1))) if count > 1 else 1
        score = np.full(length, -np.inf, dtype=np.float32)
        for _ in range(num_centres):
            centre = float(rng.uniform(0, max(length - 1, 1)))
            width = float(np.exp(rng.uniform(np.log(0.5), np.log(max(length / 3, 0.51)))))
            score = np.maximum(score, -np.abs(frame_index - centre) / width)
        score += 1e-3 * rng.normal(size=length).astype(np.float32)
        selected = _topk_mask(score, count)
    return selected.astype(np.uint8, copy=False)


def _sample_mirror_closed_subset(
    joints: np.ndarray,
    rng: np.random.RandomState,
    target_count: int,
) -> np.ndarray:
    """Sample a generic left-right symmetric subset on the SMPL tree."""
    allowed = set(int(joint) for joint in joints)
    bilateral = [
        int(joint) for joint in joints
        if int(SMPL22_MIRROR[joint]) != int(joint)
        and int(SMPL22_MIRROR[joint]) in allowed
    ]
    if bilateral and rng.random() < 0.55:
        seed = int(rng.choice(bilateral))
        return np.asarray(sorted({seed, int(SMPL22_MIRROR[seed])}), dtype=np.int64)

    selected: set[int] = set()
    while len(selected) < target_count:
        seed = int(rng.choice(joints))
        group = [seed]
        mirror = int(SMPL22_MIRROR[seed])
        if mirror in allowed and mirror != seed:
            group.append(mirror)
        remaining = target_count - len(selected)
        unseen = [joint for joint in group if joint not in selected]
        if len(unseen) <= remaining:
            selected.update(unseen)
        else:
            selected.add(int(rng.choice(unseen)))
    return np.asarray(sorted(selected), dtype=np.int64)


def _sample_tree_cut_subset(
    joints: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Sample either side of a random hierarchical cut of the kinematic tree."""
    allowed = set(int(joint) for joint in joints)
    candidates = np.asarray(
        [joint for joint in joints if SMPL22_PARENTS[joint] >= 0],
        dtype=np.int64,
    )
    if len(candidates) == 0:
        return joints.copy()

    depths = np.zeros(NUM_JOINTS, dtype=np.int64)
    for joint in range(1, NUM_JOINTS):
        depths[joint] = depths[int(SMPL22_PARENTS[joint])] + 1
    available_depths = np.unique(depths[candidates])
    depth = int(rng.choice(available_depths))
    child = int(rng.choice(candidates[depths[candidates] == depth]))

    descendants = set()
    for joint in allowed:
        current = joint
        while current >= 0:
            if current == child:
                descendants.add(joint)
                break
            current = int(SMPL22_PARENTS[current])
    complement = allowed - descendants
    chosen = descendants if rng.random() < 0.5 else complement
    if not chosen:
        chosen = descendants or complement or allowed
    return np.asarray(sorted(chosen), dtype=np.int64)


def _sample_joint_subset(
    joints: Sequence[int],
    rng: np.random.RandomState,
    *,
    root_singleton_prob: float = 0.0,
) -> np.ndarray:
    """Sample joint indices with singleton, graph-local, and random support."""
    joints = np.asarray(joints, dtype=np.int64)
    if root_singleton_prob > 0.0 and 0 in joints and rng.random() < root_singleton_prob:
        return np.asarray([0], dtype=np.int64)
    if rng.random() < 0.12:
        return joints.copy()

    mode = float(rng.random())
    if mode < 0.22:
        return _sample_tree_cut_subset(joints, rng)

    count = _sample_known_count(
        len(joints),
        rng,
        COORDINATE_DENSITY_BIN_WEIGHTS,
    )
    if mode < 0.50:
        return _sample_mirror_closed_subset(joints, rng, count)
    if mode < 0.75:
        center = int(rng.choice(joints))
        score = -GRAPH_DISTANCES[center, joints]
        score = score + 0.15 * rng.normal(size=len(joints))
        return joints[np.argpartition(score, -count)[-count:]]
    return np.asarray(rng.choice(joints, size=count, replace=False), dtype=np.int64)


def _sample_translation_atoms(rng: np.random.RandomState) -> np.ndarray:
    axes = _sample_axis_subset(rng, TRANSLATION_AXIS_SUBSET_WEIGHTS)
    return TRANSLATION_ATOMS[np.asarray(axes, dtype=np.int64)]


def _sample_rotation_atoms(
    rng: np.random.RandomState,
) -> np.ndarray:
    joints = _sample_joint_subset(
        np.arange(NUM_JOINTS),
        rng,
        root_singleton_prob=0.55,
    )
    return 3 + joints


def _sample_position_atoms(
    rng: np.random.RandomState,
) -> np.ndarray:
    joints = _sample_joint_subset(
        np.arange(1, NUM_JOINTS),
        rng,
    )
    shared_axes = _sample_axis_subset(rng) if rng.random() < 0.60 else None
    selected: List[int] = []
    for joint in joints:
        axes = shared_axes if shared_axes is not None else _sample_axis_subset(rng)
        base = 3 + NUM_JOINTS + 3 * (int(joint) - 1)
        selected.extend(base + int(axis) for axis in axes)
    return np.asarray(selected, dtype=np.int64)


def _sample_coordinate_atom(
    rng: np.random.RandomState,
) -> np.ndarray:
    """Sample representation-semantic coordinates without naming a task."""
    atom_type = int(
        rng.choice(5, p=COORDINATE_TYPE_WEIGHTS / COORDINATE_TYPE_WEIGHTS.sum())
    )
    if atom_type == 0:
        return np.arange(NUM_ATOMS, dtype=np.int64)
    if atom_type == 1:
        return _sample_translation_atoms(rng)
    if atom_type == 2:
        return _sample_rotation_atoms(rng)
    if atom_type == 3:
        return _sample_position_atoms(rng)

    # Mixed atoms compose modalities before the time-coordinate outer product.
    # This makes simultaneous heterogeneous controls common while each
    # constituent modality retains arbitrary joint/axis support.
    modalities = (
        ("translation", "rotation"),
        ("translation", "position"),
        ("rotation", "position"),
        ("translation", "rotation", "position"),
    )
    # Keep pairwise translation/rotation evidence well represented without
    # introducing a task-specific trajectory template. Three-way mixtures
    # remain frequent through both this branch and Boolean Rank-K composition.
    chosen = modalities[int(rng.choice(4, p=(0.70, 0.10, 0.10, 0.10)))]
    selected = []
    if "translation" in chosen:
        selected.append(_sample_translation_atoms(rng))
    if "rotation" in chosen:
        selected.append(_sample_rotation_atoms(rng))
    if "position" in chosen:
        selected.append(_sample_position_atoms(rng))
    return np.unique(np.concatenate(selected)).astype(np.int64, copy=False)


def _sample_boolean_rank_mask(
    length: int,
    rng: np.random.RandomState,
    *,
    density_bin_weights: Tuple[float, ...],
    max_rank: int,
) -> np.ndarray:
    """Sample ``OR_k(time_subset_k outer coordinate_subset_k)``."""
    # Rank-one masks have their own high-mass branch. This branch is reserved
    # for genuinely heterogeneous compositions so mixed evidence does not
    # remain a vanishing tail event.
    rank = _sample_rank(max_rank, rng, min_rank=min(3, max_rank))
    known = np.zeros((length, NUM_ATOMS), dtype=np.uint8)
    time_atoms: List[np.ndarray] = []
    for term in range(rank):
        if term > 0 and rng.random() < REUSE_TIME_ATOM_PROB:
            time_atom = time_atoms[int(rng.randint(len(time_atoms)))]
        else:
            time_atom = _sample_temporal_atom(length, rng, density_bin_weights)
            time_atoms.append(time_atom)
        coordinate_atom = _sample_coordinate_atom(rng)
        known[np.ix_(time_atom > 0, coordinate_atom)] = 1

    # A fully specified clip has no learning target.  Keep the sampled layout
    # and release one semantic cell rather than resampling with hidden bias.
    if known.all():
        known[int(rng.randint(length)), int(rng.randint(NUM_ATOMS))] = 0
    return known


def _sample_single_clause_mask(
    length: int,
    rng: np.random.RandomState,
    *,
    density_bin_weights: Tuple[float, ...],
) -> np.ndarray:
    """Sample one exact ``time_subset x coordinate_subset`` observation.

    Low-complexity observations are not benchmark templates: they are the
    rank-one basis elements from which arbitrary Boolean mask combinations are
    composed. Giving this basis explicit finite-sample mass is important. If
    it is reached only through the tail of the rank distribution, a nominally
    supported sparse condition can occur too rarely for the network to learn
    the dynamics on either side of its isolated observations.
    """
    time_atom = _sample_temporal_atom(length, rng, density_bin_weights)
    coordinate_atom = _sample_coordinate_atom(rng)
    known = np.zeros((length, NUM_ATOMS), dtype=np.uint8)
    known[np.ix_(time_atom > 0, coordinate_atom)] = 1
    if known.all():
        known[int(rng.randint(length)), int(rng.randint(NUM_ATOMS))] = 0
    return known


def _topk_mask(score: np.ndarray, count: int) -> np.ndarray:
    flat = score.reshape(-1)
    selected = np.zeros(flat.shape[0], dtype=np.uint8)
    indices = np.argpartition(flat, -count)[-count:]
    selected[indices] = 1
    return selected.reshape(score.shape)


def _sample_random_field_mask(
    length: int,
    rng: np.random.RandomState,
    *,
    density_bin_weights: Tuple[float, ...],
    max_rank: int,
    iid_floor: float,
) -> np.ndarray:
    """Sample arbitrary atom patterns on an explicit temporal support.

    The old implementation sampled a total number of cells from ``T x atom``.
    As a result, temporal density was an uncontrolled consequence of score
    thresholding. Here the active frames are sampled first, and the random field
    only decides which semantic atoms are known on those frames.
    """
    time_atom = _sample_temporal_atom(length, rng, density_bin_weights)
    active_frames = np.flatnonzero(time_atom)
    score = _sample_score_grid(
        length,
        rng,
        max_rank=max_rank,
        iid_floor=iid_floor,
    )

    active_score = score[active_frames]
    active_size = int(active_score.size)
    known_count = _sample_known_count(
        active_size,
        rng,
        COORDINATE_DENSITY_BIN_WEIGHTS,
    )
    # Each temporally selected frame must contain condition evidence. This makes
    # the realized frame density exactly equal to the sampled temporal density.
    known_count = max(known_count, len(active_frames))
    known_active = _topk_mask(active_score, known_count)
    empty_rows = np.flatnonzero(~known_active.any(axis=1))
    for row in empty_rows:
        known_active[row, int(np.argmax(active_score[row]))] = 1

    known = np.zeros((length, NUM_ATOMS), dtype=np.uint8)
    known[active_frames] = known_active
    return known


def semantic_known_to_dim_mask(known_atoms: np.ndarray) -> np.ndarray:
    """Expand semantic known atoms to a ``(T, 198)`` generate mask."""
    length = known_atoms.shape[0]
    mask = np.ones((length, MOTION_DIM), dtype=np.float32)
    for atom_index, atom in enumerate(ATOMS):
        active = known_atoms[:, atom_index] > 0
        if active.any():
            mask[np.ix_(active, atom.dims)] = 0.0
    return mask


def sample_mask(
    length: int,
    rng: np.random.RandomState,
    *,
    pure_generation_prob: float = 0.05,
    density_bin_weights: Tuple[float, ...] = (0.30, 0.25, 0.18, 0.14, 0.13),
    max_rank: int = 8,
    iid_floor: float = 0.02,
) -> np.ndarray:
    """Sample one arbitrary condition-target mask without task templates."""
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if not 0.0 <= pure_generation_prob < 1.0:
        raise ValueError("pure_generation_prob must be in [0, 1)")
    if max_rank <= 0:
        raise ValueError("max_rank must be positive")
    if rng.random() < pure_generation_prob:
        return np.ones((length, MOTION_DIM), dtype=np.float32)

    sampler_draw = float(rng.random())
    if sampler_draw < RANDOM_FIELD_PROB:
        known_atoms = _sample_random_field_mask(
            length,
            rng,
            density_bin_weights=density_bin_weights,
            max_rank=max_rank,
            iid_floor=iid_floor,
        )
    elif sampler_draw < RANDOM_FIELD_PROB + SINGLE_CLAUSE_PROB:
        known_atoms = _sample_single_clause_mask(
            length,
            rng,
            density_bin_weights=density_bin_weights,
        )
    else:
        known_atoms = _sample_boolean_rank_mask(
            length,
            rng,
            density_bin_weights=density_bin_weights,
            max_rank=max_rank,
        )
    return semantic_known_to_dim_mask(known_atoms)


def sample_condition(
    length: int,
    rng: np.random.RandomState,
    **kwargs,
) -> Tuple[np.ndarray, bool]:
    """Return the final sampler contract used by condition preparation."""
    return sample_mask(length, rng, **kwargs), False
