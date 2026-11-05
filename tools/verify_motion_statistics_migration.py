"""Compare relocated data transforms with an explicit pre-migration snapshot.

This is a migration regression certificate, not an independent upstream result.
"""

from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from motius.motion.representation import normalization as transforms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = dict(
        complete=False,
        scope=__doc__,
        snapshot=str(args.snapshot.resolve()),
        comparisons=[],
    )
    torch.set_num_threads(2)
    torch.manual_seed(471)
    names = {
        "normalize",
        "denormalize",
        "normalize_absolute",
        "denormalize_absolute",
        "normalize_joints",
        "denormalize_joints",
    }
    for path in sorted(args.snapshot.glob("*.py")):
        original = ast.parse(path.read_text())
        functions = [
            node
            for item in original.body
            if isinstance(item, ast.ClassDef)
            for node in item.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        namespace = dict(torch=torch, np=np)
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *functions,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        for dtype in [torch.float32, torch.float64, torch.bfloat16]:
            for shape in [(9, 38), (2, 7, 263)]:
                dimension = shape[-1]
                mean = torch.randn(dimension)
                std = torch.rand(dimension) + 0.01
                bundle = SimpleNamespace(
                    mean=mean,
                    std=std,
                    mean_abs=mean,
                    std_abs=std,
                    joints_mean=mean,
                    joints_std=std,
                    normalizer_min=mean,
                    normalizer_scale=std,
                    device=torch.device("cpu"),
                )
                motion = torch.randn(shape).to(dtype)
                for function in functions:
                    normalizing = function.name.startswith("normalize")
                    target = (
                        transforms.affine_normalize
                        if normalizing
                        else transforms.affine_denormalize
                    )
                    if path.stem == "flowmdm":
                        target = transforms.matched_denormalize
                    if path.stem == "edge":
                        target = transforms.range_denormalize
                    if path.stem == "motionclr":
                        target = (
                            transforms.array_normalize
                            if normalizing
                            else transforms.array_denormalize
                        )
                    candidates = [motion]
                    if path.stem == "motionclr" and dtype != torch.bfloat16:
                        candidates.append(motion.numpy())
                    for value in candidates:
                        expected = namespace[function.name](bundle, value)
                        actual = target(value, mean, std)
                        if isinstance(expected, np.ndarray):
                            assert np.array_equal(expected, actual)
                        else:
                            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                            assert expected.dtype == actual.dtype
                        report["comparisons"].append(
                            dict(
                                method=path.stem,
                                function=function.name,
                                dtype=str(dtype),
                                shape=shape,
                                input_type=type(value).__name__,
                                exact=True,
                            )
                        )
    report["complete"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Exact migration comparisons:", len(report["comparisons"]))


if __name__ == "__main__":
    main()
