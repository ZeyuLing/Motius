#!/usr/bin/env python3
"""Replay native character control against unmodified official TensorRT/character code.

The scene dispatcher is exercised by native smoke tests; this independent oracle
covers character states, target blending, recording, generation and playback.
Reference-only assets and TensorRT plans remain outside the installed package.
"""

import argparse
import importlib
import json
import random
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import warp as wp

from motius import Pipeline


def torch_reference_engine(engine_cls, rig, sources_root, transform):
    """Only replace transport/backend; inherit all official generation geometry."""
    from motionbricks.model.generation import GenerationTimer
    from motionbricks_reference.checkpoints import checkpoint_root, load_sources
    from motionbricks_reference.checksums import CHECKPOINTS

    network_module = importlib.import_module(
        "motionbricks_reference.models." + rig + ".model"
    )
    release = CHECKPOINTS[rig]
    network = getattr(network_module, "build_" + rig)(
        load_sources(checkpoint_root(sources_root / rig, release), release)
    ).cuda()
    for module in network.modules():
        if hasattr(module, "_device"):
            module._device = torch.device("cuda:0")
    for name in ("capture_offsets", "standard_offsets"):
        if hasattr(network, name):
            setattr(network, name, getattr(network, name).cuda())

    class Reference(engine_cls):
        def __init__(self):
            self._closed = False
            self._stream = wp.stream_from_torch(torch.cuda.current_stream())
            self._use_cuda_graph = False
            self._model_to_runtime = transform
            self._runtime_to_model = wp.transform_inverse(transform)
            self.timer = GenerationTimer(self.stream)
            self._alignment = wp.empty(1, dtype=wp.transform, device=self.device)
            self._generated_frame_count = wp.empty(
                1, dtype=wp.int32, device=self.device
            )
            self.tensors = {
                spec.name: torch.empty(
                    spec.shape,
                    device="cuda",
                    dtype=torch.from_numpy(np.empty((), dtype=spec.dtype)).dtype,
                )
                for spec in self.contract.tensor_specs
            }
            self._buffers = {k: wp.from_torch(v) for k, v in self.tensors.items()}
            self._initialize_pose_layout()

        def enqueue_async(self):
            # Verification adapter: finish Warp writes before handing shared buffers to Torch.
            wp.synchronize_device(self.device)
            torch.cuda.synchronize()
            with (
                torch.inference_mode(),
                torch.cuda.stream(wp.stream_to_torch(self.stream)),
            ):
                values = network(
                    **{k: self.tensors[k] for k in network_module.INPUT_NAMES}
                )
                for k, v in zip(network_module.OUTPUT_NAMES, values):
                    self.tensors[k].copy_(v)
            torch.cuda.synchronize()

        def close(self):
            self.synchronize()
            self._closed = True

    return Reference()


def run(
    source,
    assets,
    checkpoints,
    cache_dir,
    backend="pytorch",
    sources_root=Path("checkpoints/models/motionbricks_safetensors"),
):
    from tools.verify_motionbricks_release_parity import reference_package

    reference_package(source)  # Assert the pinned source revision.
    sys.path.insert(0, str(source))
    # Hydrate reference data, never change reference Python source.
    for folder in ("motions", "smart_primitives", "scenes"):
        shutil.copytree(assets / folder, source / "assets" / folder, dirs_exist_ok=True)
    from app.runtime.common.clip_registry import ClipRegistry
    from app.runtime.common.io.scene_document import load_scene_document
    from app.runtime.common.io.smart_primitive_document import SmartPrimitiveAssetPool
    from app.runtime.common.motion import LocomotionCommand as RefCommand
    from app.runtime.common.rig import model_to_runtime
    from app.runtime.common.smart_primitive_behaviors import behavior_types
    from motionbricks.inference.builder import TensorRTBuildOptions
    from motionbricks.model.contract import ModelRig
    from motionbricks.model.unitree_g1.contract import UNITREE_G1_N1_CONTRACT

    from motius.pipelines.motionbricks.scene_motion import LocomotionCommand

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    wp.init()
    wp.set_device("cuda:0")
    results = []
    rng_type = random.Random
    for rig in ("g1", "soma"):
        folder = "unitree_g1" if rig == "g1" else "soma"
        rig_enum = ModelRig.UNITREE_G1 if rig == "g1" else ModelRig.SOMA
        module = importlib.import_module("motionbricks.model." + folder + ".engine")
        engine_cls = (
            module.MotionBricksUnitreeG1Engine
            if rig == "g1"
            else module.MotionBricksSOMAEngine
        )
        if backend == "pytorch":
            reference = torch_reference_engine(
                engine_cls, rig, sources_root, model_to_runtime(rig_enum)
            )
        else:
            reference = engine_cls(
                source / f"motionbricks/assets/onnx/{folder}/mmm_v6_{rig}_fp32.onnx",
                model_to_runtime=model_to_runtime(rig_enum),
                build_options=TensorRTBuildOptions(
                    enable_tf32=False,
                    workspace_size_bytes=512 * 1024**2,
                    builder_optimization_level=0,
                ),
                cache_dir=cache_dir,
                bake_cuda_graph=False,
            )
        wp.set_stream(reference.stream)
        characters = importlib.import_module("app.runtime." + folder + ".character")
        character_cls = (
            characters.UnitreeG1Character if rig == "g1" else characters.SOMACharacter
        )
        if rig == "soma":
            from motionbricks.io.usd import load_skeleton_from_usd

            skel = load_skeleton_from_usd(
                assets / "human/soma_base_skel_minimal_no_root.usd",
                "/OUTPUT/c_skeleton_grp/Hips",
            )
            registry = ClipRegistry(rig_enum, skeleton=skel)
            pose = skel.reference_local_transforms
        else:
            registry = ClipRegistry(
                rig_enum,
                joint_names=UNITREE_G1_N1_CONTRACT.joint_names,
                device=reference.device,
            )
            from motionbricks.io.utils import get_unitree_g1_model

            reference_robot = get_unitree_g1_model(
                device=reference.device, load_visuals=False
            )
            pose = wp.clone(reference_robot.joint_q)
        pool = SmartPrimitiveAssetPool(registry, pose, model_to_runtime(rig_enum))
        doc, _, _ = load_scene_document(
            source / "assets/scenes" / f"{rig}_gym.scene.json"
        )
        doc = doc.flatten()
        with patch("random.Random", lambda *args: rng_type(42)):
            character = character_cls(
                reference,
                asset_pool=pool,
                initial_transform=doc.character_start.transform,
            )
        primitives = []
        for item in doc.smart_primitives:
            primitive = pool.instantiate(
                item.reference, behavior_types()[item.behavior]
            )
            primitive.set_node_world_transform(
                0, item.transform, stream=reference.stream
            )
            primitives.append(primitive)
        pipe = Pipeline.from_pretrained(
            checkpoints / rig, bundle_kwargs={"device": "cuda"}
        )
        with pipe.load_scene(
            assets / "scenes" / f"{rig}_gym.scene.json", assets_root=assets, seed=42
        ) as scene:
            phases = [
                (None, False, 30),
                ("linear_1d", True, 90),
                ("directional_2d", True, 90),
                ("crouch_directional_2d", True, 90),
                ("linear_1d", False, 60),
                ("zombie_1d", True, 60),
            ]
            peak = 0.0
            frames = 0
            states = set()

            def advance(
                mode,
                moving,
                n,
                *,
                reference=reference,
                character=character,
                scene=scene,
                rig=rig,
                states=states,
            ):
                nonlocal peak, frames
                for i in range(n):
                    kw = dict(
                        facing_direction=np.array([1.0, 0.0, 0.0], np.float32),
                        movement=np.array([1.0, 0.0, 0.0], np.float32)
                        if moving
                        else None,
                        requested_mode=mode if i == 0 else None,
                    )
                    actual = scene.step(LocomotionCommand(**kw))
                    with wp.ScopedStream(reference.stream):
                        character.step(1 / 30, RefCommand(**kw))
                    reference.synchronize()
                    a = actual["pose"]
                    b = character.current_pose.numpy()
                    if backend == "pytorch":
                        np.testing.assert_array_equal(
                            a, b, err_msg=f"{rig} frame {frames}"
                        )
                    qa = a[..., 3:7]
                    qb = b[..., 3:7]
                    qb *= np.where((qa * qb).sum(-1, keepdims=True) < 0, -1, 1)
                    error = float(np.max(np.abs(a - b)))
                    peak = max(peak, error)
                    assert actual["state"] == character.state_machine.state.name, (
                        frames,
                        actual["state"],
                        character.state_machine.state.name,
                    )
                    np.testing.assert_allclose(
                        a, b, atol=2e-3, rtol=2e-3, err_msg=f"{rig} frame {frames}"
                    )
                    states.add(actual["state"])
                    frames += 1

            for mode, moving, n in phases:
                advance(mode, moving, n)
                print(rig, mode, frames, peak, flush=True)
            interactions = []
            for placed, primitive in zip(scene.primitives, primitives):
                if scene.state_machine.is_interacting:
                    scene.interact()
                    character.state_machine.interact()
                    advance(None, False, 90)
                if scene.state_machine.is_interacting:
                    raise AssertionError("Previous interaction did not finish")
                scene.interact(placed.definition.id)
                character.state_machine.start_interaction(character, primitive)
                advance(None, False, 120)
                interactions.append(placed.definition.behavior)
                print(
                    rig,
                    "interaction",
                    placed.definition.behavior,
                    frames,
                    peak,
                    flush=True,
                )
            results.append(
                dict(
                    rig=rig,
                    bitwise_equal=(backend == "pytorch" and peak == 0),
                    frames=frames,
                    pose_max_abs_error=peak,
                    states=sorted(states),
                    interactions=interactions,
                )
            )
        reference.close()
        del pipe, reference, character, pool
        torch.cuda.empty_cache()
    return dict(
        complete=True,
        source_revision="54fc563aeaaf819a753d72cc9fdc6713989cdd43",
        device=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        warp_version=wp.__version__,
        atol=0 if backend == "pytorch" else 2e-3,
        rtol=0 if backend == "pytorch" else 2e-3,
        backend=backend,
        scope="Independent official model and character/state/playback code, scripted controls and primitive interactions; excludes renderer/UI and automatic nearest-primitive dispatcher",
        tests=results,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--backend", choices=["pytorch", "tensorrt"], default="pytorch")
    p.add_argument(
        "--sources-root",
        type=Path,
        default=Path("checkpoints/models/motionbricks_safetensors"),
    )
    p.add_argument(
        "--assets",
        type=Path,
        default=Path("checkpoints/models/motionbricks_scene_assets").resolve(),
    )
    p.add_argument(
        "--checkpoints",
        type=Path,
        default=Path("checkpoints/models/motionbricks_native"),
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs/motionbricks_native_audit/trt_cache"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("docs/architecture/validation/motionbricks-scene-cuda.json"),
    )
    a = p.parse_args()
    result = run(
        a.source, a.assets, a.checkpoints, a.cache_dir, a.backend, a.sources_root
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
