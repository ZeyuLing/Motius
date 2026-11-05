"""Headless scene control over native MotionBricks generation.

Loads official scene/primitive documents as data. Rendering, input devices and
Newton simulation are not prerequisites for the character's motion policy.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import warp as wp

from motius.configuration.motionbricks_runtime_contract import (
    UNITREE_G1_N1_CONTRACT,
    ModelRig,
)
from motius.motion.io.motionbricks.scene_document import (
    check_reference_rigs,
    load_scene_document,
)
from motius.motion.io.motionbricks.smart_primitive_document import (
    SmartPrimitiveAssetPool,
)

from .scene_assets import model_to_runtime
from .scene_clip_registry import ClipRegistry
from .scene_motion import LocomotionCommand
from .scene_smart_primitive import InteractionBox, compute_interaction_box_overlaps
from .scene_smart_primitive_behaviors import behavior_types


@dataclass
class PlacedPrimitive:
    definition: object
    runtime: object
    offset: int
    interactable: bool = False

    def distance_squared(self, position):
        delta = np.asarray(self.definition.transform.p, dtype=np.float32) - position
        return float(delta @ delta)


class MotionBricksScene:
    def __init__(
        self, pipeline, scene_path, *, assets_root, reference_clip=None, seed=None
    ):
        self.assets_root = Path(assets_root).expanduser().resolve()
        definition, _, _ = load_scene_document(scene_path)
        check_reference_rigs(definition)
        self.definition = definition.flatten()
        rig = ModelRig.SOMA if pipeline.bundle.model == "soma" else ModelRig.UNITREE_G1
        if rig != definition.rig:
            raise ValueError("Scene rig does not match the pipeline checkpoint")
        self.engine = pipeline.create_runtime(model_to_runtime=model_to_runtime(rig))
        try:
            with (
                wp.ScopedDevice(self.engine.device),
                wp.ScopedStream(self.engine.stream),
            ):
                if rig is ModelRig.SOMA:
                    from motius.motion.io.motionbricks.bvh import load_bvh

                    from . import scene_soma_presets as presets
                    from .scene_soma_character import SOMACharacter as Character

                    source = (
                        Path(reference_clip)
                        if reference_clip is not None
                        else self.assets_root
                        / "motions/soma/locomotion/idle/idle_loop_001__A268.bvh"
                    )
                    from motius.motion.io.motionbricks.usd_skeleton import (
                        load_skeleton_from_usd,
                    )

                    skeleton = load_skeleton_from_usd(
                        self.assets_root / "human/soma_base_skel_minimal_no_root.usd",
                        "/OUTPUT/c_skeleton_grp/Hips",
                    )
                    clip = load_bvh(source, skeleton)
                    registry = ClipRegistry(rig, skeleton=clip.skeleton)
                    reference = clip.skeleton.reference_local_transforms
                else:
                    from . import scene_unitree_g1_presets as presets
                    from .scene_unitree_g1_character import (
                        UnitreeG1Character as Character,
                    )

                    registry = ClipRegistry(
                        rig,
                        joint_names=UNITREE_G1_N1_CONTRACT.joint_names,
                        device=self.engine.device,
                    )
                    if reference_clip is not None:
                        reference = registry.load(Path(reference_clip)).animation.poses[
                            0
                        ]
                    else:
                        # Official G1 MJCF default joint_q (root at 0.793 m, all hinges zero).
                        # This is a rest pose, not the first frame of an idle clip.
                        values = np.zeros(36, dtype=np.float32)
                        values[2], values[6] = 0.793, 1.0
                        reference = wp.array(
                            values, dtype=wp.float32, device=self.engine.device
                        )
                self.asset_pool = SmartPrimitiveAssetPool(
                    registry, reference, model_to_runtime(rig)
                )
                self.asset_pool.assets_root = self.assets_root
                self.asset_pool.random_seed = seed
                modes = tuple(
                    replace(
                        mode,
                        smart_primitive_path=self.assets_root
                        / mode.smart_primitive_path,
                    )
                    for mode in presets.DEFAULT_LOCOMOTION_MODES
                )
                self.character = Character(
                    self.engine,
                    asset_pool=self.asset_pool,
                    initial_transform=self.definition.character_start.transform,
                    modes=modes,
                    idle_paths=tuple(
                        self.assets_root / p for p in presets.DEFAULT_IDLE_CLIPS
                    ),
                    crouch_idle_paths=tuple(
                        self.assets_root / p for p in presets.DEFAULT_CROUCH_IDLE_CLIPS
                    ),
                )
                self.primitives = []
                offset = 0
                for item in self.definition.smart_primitives:
                    if item.behavior not in behavior_types():
                        raise ValueError(
                            f"Unknown Smart Primitive behavior {item.behavior!r}"
                        )
                    primitive = self.asset_pool.instantiate(
                        item.reference, behavior_types()[item.behavior]
                    )
                    primitive.set_node_world_transform(
                        0, item.transform, stream=self.engine.stream
                    )
                    self.primitives.append(PlacedPrimitive(item, primitive, offset))
                    offset += primitive.interaction_box_count
                self.boxes = wp.empty(
                    max(1, offset), dtype=InteractionBox, device=self.engine.device
                )
                self.overlaps = wp.zeros(
                    max(1, offset), dtype=wp.int32, device=self.engine.device
                )
                self.box_count = offset
                self._active = None
                self._auto_blocked = set()
                self.time = 0.0
                self._refresh_interactions()
        except Exception:
            self.engine.close()
            raise

    @property
    def state_machine(self):
        return self.character.state_machine

    def _refresh_interactions(self):
        if not self.box_count:
            self.nearest = None
            return
        for placed in self.primitives:
            end = placed.offset + placed.runtime.interaction_box_count
            if end > placed.offset:
                placed.runtime.get_world_interaction_boxes(
                    self.boxes[placed.offset : end], stream=self.engine.stream
                )
        compute_interaction_box_overlaps(
            self.boxes[: self.box_count],
            self.character.probe_box,
            self.overlaps[: self.box_count],
            self.engine.stream,
        )
        self.engine.synchronize()
        overlaps = self.overlaps.numpy()
        for placed in self.primitives:
            placed.interactable = bool(
                np.any(
                    overlaps[
                        placed.offset : placed.offset
                        + placed.runtime.interaction_box_count
                    ]
                )
            )
        self.nearest = min(
            (p for p in self.primitives if p.interactable),
            key=lambda p: p.distance_squared(self.character.character_ground_position),
            default=None,
        )

    def interact(self, primitive_id=None):
        """Press interaction, or explicitly select a scene primitive for scripted control."""
        with wp.ScopedDevice(self.engine.device), wp.ScopedStream(self.engine.stream):
            self._interact(primitive_id)
            self.engine.synchronize()

    def _interact(self, primitive_id=None):
        if self.state_machine.is_interacting:
            self.state_machine.interact()
            return
        placed = (
            self.nearest
            if primitive_id is None
            else next(
                (p for p in self.primitives if p.definition.id == primitive_id), None
            )
        )
        if placed is None:
            if primitive_id is not None:
                raise ValueError(f"Unknown primitive id {primitive_id}")
            return
        self.state_machine.start_interaction(self.character, placed.runtime)
        self._active = placed

    def step(
        self, command=None, *, delta_time=1 / 30, interact=False, auto_interact=False
    ):
        if not np.isfinite(delta_time) or delta_time <= 0:
            raise ValueError("delta_time must be finite and positive")
        command = command or LocomotionCommand(
            np.array([1.0, 0.0, 0.0], dtype=np.float32), None
        )
        with wp.ScopedDevice(self.engine.device), wp.ScopedStream(self.engine.stream):
            self.character.step(delta_time, command)
            self.time += delta_time
            self._refresh_interactions()
            if self._active is not None and not self.state_machine.is_interacting:
                self._auto_blocked.add(self._active.definition.id)
                self._active = None
            for placed in self.primitives:
                if not placed.interactable:
                    self._auto_blocked.discard(placed.definition.id)
            if self.state_machine.is_interacting:
                if interact:
                    self.state_machine.interact()
            elif self.nearest is not None and (
                interact
                or (
                    auto_interact
                    and self.nearest.definition.id not in self._auto_blocked
                )
            ):
                self.interact()
            self.engine.synchronize()
        return dict(
            pose=self.character.current_pose.numpy().copy(),
            state=self.state_machine.state.name,
            mode=self.state_machine.active_locomotion_state.active_mode.name,
            time=self.time,
            nearest_primitive=None
            if self.nearest is None
            else self.nearest.definition.id,
        )

    def close(self):
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
