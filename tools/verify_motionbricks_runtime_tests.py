#!/usr/bin/env python3
"""Reproduce the selected official headless runtime regression suite."""

import argparse
import ast
import hashlib
import json
import pathlib
import subprocess
import sys
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(
    description="Run selected official algorithm tests with only import/patch namespace adaptation; excludes viewer/mesh/TRT transport tests."
)
parser.add_argument("--source", type=pathlib.Path, required=True)
parser.add_argument(
    "--work",
    type=pathlib.Path,
    default=pathlib.Path("outputs/motionbricks_native_audit/official_tests"),
)
parser.add_argument(
    "--output",
    type=pathlib.Path,
    default=pathlib.Path(
        "docs/architecture/validation/motionbricks-runtime-tests.json"
    ),
)
args = parser.parse_args()
src = args.source
out = args.work
out.mkdir(parents=True, exist_ok=True)
revision = subprocess.check_output(
    ["git", "-C", str(src), "rev-parse", "HEAD"], text=True
).strip()
assert revision == "54fc563aeaaf819a753d72cc9fdc6713989cdd43", revision
m = json.loads(
    pathlib.Path(
        "docs/architecture/validation/motionbricks-runtime-source-map.json"
    ).read_text()
)["components"]
m.update(
    {
        "motionbricks.io.usd": "motius.motion.io.motionbricks.usd_skeleton",
        "motionbricks.model.contract": "motius.configuration.motionbricks_runtime_contract",
        "motionbricks.model.soma.contract": "motius.configuration.motionbricks_runtime_contract",
        "motionbricks.model.unitree_g1.contract": "motius.configuration.motionbricks_runtime_contract",
        "motionbricks.inference.errors": "motius.configuration.motionbricks_runtime_contract",
    }
)


class Adapt(ast.NodeTransformer):
    def visit_FunctionDef(self, n):
        if n.name.startswith("test_scene_preview_"):
            return None  # GUI/viewer tests are outside the headless runtime.
        return self.generic_visit(n)

    def visit_ImportFrom(self, n):
        if n.module == "app.scene_preview":
            return None
        if n.module in m:
            n.module = m[n.module]
            return n
        result = []
        for a in n.names:
            key = (n.module or "") + "." + a.name
            if key in m:
                parent, child = m[key].rsplit(".", 1)
                result.append(
                    ast.ImportFrom(
                        module=parent,
                        names=[ast.alias(name=child, asname=a.asname or a.name)],
                        level=0,
                    )
                )
            else:
                result.append(ast.ImportFrom(module=n.module, names=[a], level=n.level))
        return result

    def visit_Import(self, n):
        for a in n.names:
            if a.name in m and a.asname:
                a.name = m[a.name]
        return n


files = [
    "motionbricks/test_" + s + ".py"
    for s in [
        "dof_endpoint_correction",
        "dof_buffers",
        "math_utils",
        "motion_player",
        "blend_spaces",
        "target_pose",
        "pose_buffers",
        "pivot",
        "animation_blending",
    ]
] + [
    "app/test_" + s + ".py"
    for s in [
        "state_machine",
        "locomotion_character_states",
        "clip_registry",
        "runtime_smart_primitive",
    ]
]
for f in files:
    text = ast.unparse(
        ast.fix_missing_locations(
            Adapt().visit(ast.parse((src / "tests" / f).read_text()))
        )
    )
    for old, new in m.items():
        text = text.replace(old + ".", new + ".")
    (out / pathlib.Path(f).name).write_text(text + "\n")
xml = out / "results.xml"
subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        *[str(out / pathlib.Path(f).name) for f in files],
        "-q",
        "--disable-warnings",
        "--junitxml=" + str(xml),
    ],
    check=True,
)
suites = ET.parse(xml).getroot().findall("testsuite")
report = dict(
    complete=True,
    source_revision=revision,
    scope="Official algorithm tests, assertions unchanged; imports and mock patch namespaces adapted to native modules",
    excluded_test_prefixes=["test_scene_preview_"],
    tests=sum(int(x.attrib["tests"]) for x in suites),
    failures=sum(int(x.attrib["failures"]) for x in suites),
    errors=sum(int(x.attrib["errors"]) for x in suites),
    skipped=sum(int(x.attrib["skipped"]) for x in suites),
    source_files={
        f: hashlib.sha256((src / "tests" / f).read_bytes()).hexdigest() for f in files
    },
)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
