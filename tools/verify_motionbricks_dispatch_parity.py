#!/usr/bin/env python3
"""Exhaustively compare headless interaction dispatch with pinned official source."""

import argparse
import ast
import itertools
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import warp as wp
from motius.pipelines.motionbricks.scene import MotionBricksScene
from tools.verify_motionbricks_release_parity import reference_package


def run(source):
    reference_package(source)
    tree = ast.parse((source / "app/scene_preview.py").read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ScenePreview"
    )
    selected = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {"_start_smart_primitive_interaction", "_dispatch_interaction_input"}
    ]
    cls.body = selected
    cls.bases = []
    cls.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(module),
            str(source / "app/scene_preview.py"),
            "exec",
        ),
        namespace,
    )
    Oracle = namespace["ScenePreview"]

    def machine(active):
        result = NS(
            is_interacting=active,
            events=[],
            state=NS(name="test"),
            active_locomotion_state=NS(active_mode=NS(name="test")),
        )

        def start(character, runtime):
            result.events.append("start")
            result.is_interacting = True

        result.start_interaction = start
        result.interact = lambda: result.events.append("press")
        return result

    count = 0
    for active, previous, near, blocked, pressed, automatic in itertools.product(
        (False, True), repeat=6
    ):
        primitive = NS(definition=NS(id=7), runtime=object(), interactable=near)
        native = MotionBricksScene.__new__(MotionBricksScene)
        native.engine = NS(device="cpu", stream=None, synchronize=lambda: None)
        native.character = NS(
            state_machine=machine(active),
            current_pose=wp.array(np.zeros(36, np.float32), device="cpu"),
            step=lambda *args: None,
        )
        native.primitives = [primitive]
        native.nearest = primitive if near else None
        native._active = primitive if previous else None
        native._auto_blocked = {7} if blocked else set()
        native.time = 0
        native._refresh_interactions = lambda: None
        oracle = Oracle.__new__(Oracle)
        oracle._scene_session = NS(
            state_machine=machine(active),
            character=object(),
            smart_primitives=[primitive],
        )
        oracle._active_interaction = primitive if previous else None
        oracle._nearest_interactable = primitive if near else None
        oracle._auto_interaction_blocked = {7} if blocked else set()
        oracle._interaction_pressed = lambda: pressed
        oracle.auto_interaction = automatic
        native.step(interact=pressed, auto_interact=automatic)
        oracle._dispatch_interaction_input()
        assert native.state_machine.events == oracle._scene_session.state_machine.events
        assert native._auto_blocked == oracle._auto_interaction_blocked
        assert (native._active is None) == (oracle._active_interaction is None)
        count += 1
    return dict(
        complete=True,
        cases=count,
        source_revision="54fc563aeaaf819a753d72cc9fdc6713989cdd43",
        scope="Manual/automatic dispatch, active interaction button forwarding, completion blocking and leave/re-enter unblocking; all boolean combinations against unchanged official methods",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("docs/architecture/validation/motionbricks-dispatch.json"),
    )
    a = p.parse_args()
    result = run(a.source)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result)
