"""Check that a wheel ships current package files and no stale application cache."""

from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def audit(wheel, root=ROOT):
    findings = []
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        python = [
            name
            for name in names
            if name.startswith("motius/") and name.endswith(".py")
        ]
        for name in sorted(python):
            path = root / name
            if not path.is_file():
                findings.append(dict(kind="stale_python", path=name))
            elif archive.read(name) != path.read_bytes():
                findings.append(dict(kind="outdated_python", path=name))
        for name in names:
            if "__pycache__" in name or name.endswith((".pyc", ".pyo")):
                findings.append(dict(kind="bytecode_cache", path=name))
            if (
                name.startswith(("motius/models/", "motius/trainers/"))
                and "/vendor/" in name
            ):
                findings.append(dict(kind="vendor_application", path=name))
        for path in (root / "motius").rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            # setuptools namespaces=False: only regular Python packages belong here.
            parent = path.parent
            while parent != root and (parent / "__init__.py").is_file():
                parent = parent.parent
            if parent == root and str(path.relative_to(root)) not in names:
                findings.append(
                    dict(kind="missing_python", path=str(path.relative_to(root)))
                )
        required = (
            [
                f"motius/motion/assets/omnicontrol/{name}.npy"
                for name in ["Mean", "Std", "Mean_raw", "Std_raw"]
            ]
            + [
                f"motius/motion/assets/projflow/joints_{name}.npy"
                for name in ["mean", "std"]
            ]
            + [
                f"motius/licenses/{method}/{name}"
                for method in ["mdm", "condmdi", "omnicontrol"]
                for name in ["LICENSE", "CLIP_LICENSE", "ATTRIBUTIONS.md"]
            ]
        )
        required += [
            "motius/licenses/dart/PYTORCH3D_LICENSE",
            "motius/licenses/dart/PYTORCH3D_PROVENANCE.json",
            "motius/licenses/humanml3d_evaluator/LICENSE",
            "motius/licenses/humanml3d_evaluator/ATTRIBUTIONS.md",
            "motius/licenses/hymotion_t2m/UPSTREAM_LICENSE.md",
            "motius/licenses/hymotion_t2m/WOODEN_ASSETS.json",
            "motius/licenses/edge/LICENSE",
            "motius/licenses/edge/ATTRIBUTIONS.md",
            "motius/licenses/edge/PYTORCH3D_LICENSE",
            "motius/licenses/mld/LICENSE",
            "motius/licenses/mld/CLIP_LICENSE",
            "motius/licenses/mld/ATTRIBUTIONS.md",
            "motius/licenses/motionlcm/UPSTREAM_LICENSE.md",
            "motius/licenses/motionlcm/ATTRIBUTIONS.md",
        ]
        required += [
            f"motius/motion/assets/hymotion_wooden/{name}.bin"
            for name in [
                "v_template",
                "j_template",
                "skinWeights",
                "skinIndice",
                "kintree",
            ]
        ]
        for name in required:
            if name not in names:
                findings.append(dict(kind="missing_runtime_data", path=name))
            elif archive.read(name) != (root / name).read_bytes():
                findings.append(dict(kind="outdated_runtime_data", path=name))
        for path in (root / "motius/licenses").rglob("*"):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path.suffix == ".py"
            ):
                continue
            name = str(path.relative_to(root))
            if name not in names:
                findings.append(dict(kind="missing_source_notice", path=name))
            elif archive.read(name) != path.read_bytes():
                findings.append(dict(kind="outdated_source_notice", path=name))
    return dict(
        complete=not findings,
        wheel=str(wheel),
        wheel_sha256=hashlib.sha256(Path(wheel).read_bytes()).hexdigest(),
        python_files=len(python),
        findings=findings,
        scope="Archive files versus current source, regular-package completeness, vendor applications and selected relocated runtime statistics/legal notices. Does not establish runtime optional-dependency closure or numerical source parity.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.wheel, args.root)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
