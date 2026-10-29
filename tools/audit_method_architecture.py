"""Inventory responsibility leaks in method implementations without importing them.

This reports evidence, not a numerical-parity certificate. A method only becomes
native after its remaining findings are reviewed and its source comparison passes.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import warnings


ROOT = Path(__file__).resolve().parents[1]
NON_NETWORK_PARTS = {
    "vendor", "datasets", "data_loaders", "trainer", "trainers", "evaluation",
    "evaluators", "visualization", "visualize", "rendering", "pipeline", "pipelines",
}
NON_NETWORK_FILES = {
    "inference.py", "sampler.py", "evaluator.py", "preprocess.py", "audio.py",
    "collate.py", "editing.py", "representation.py", "reward.py", "dataset.py",
    "processor.py", "tokenizer.py", "word_vectorizer.py",
    "text_encoder_api.py", "ardy_model.py", "keypoint_prompt_sampler.py",
    "m2m_loss.py", "smpl_lite.py", "_smoothing.py",
    "task_instruction.py", "checkpoint_loading.py",
}
ORCHESTRATION_METHODS = {
    "train_step", "val_step", "training_step", "validation_step", "configure_optimizers",
    "compute_loss", "training_forward", "forward_in_training", "generate_from_feature",
    "grpo_sample", "grpo_one_step", "p_sample_loop", "ddim_sample_loop",
    "sample_cached", "sample_batch", "forward_with_cfg", "forward_with_cond_scale",
    "generate", "generate_codes", "generate_control", "edit", "edit_beta", "forward_loss",
    "_build_scheduler", "assemble_observation", "decode_motor_targets",
    "decode_motion_from_latent", "encode_primary_body", "_one_prompt_iter", "data_preprocess", "normalize_motion", "denormalize_motion",
    "normalize_absolute", "denormalize_absolute", "normalize_joints", "denormalize_joints",
    "normalize", "denormalize", "unnormalize",
    "denoising_step", "autoregressive_step", "_generate_window",
    "guided_forward", "make_beta_schedule",
    "prepare_padding", "prepare_condition_context", "maybe_inject_source_token", "mask_text_cond",
}


# These construct positional encodings, attention masks or learned normalization layers.
ARCHITECTURE_GENERATORS = {"generate_src_mask", "generate_pos_embs_mix2d", "Normalize"}


def is_main_guard(node):
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    values = [test.left, test.comparators[0]]
    return any(isinstance(value, ast.Name) and value.id == '__name__' for value in values) and any(
        isinstance(value, ast.Constant) and value.value == '__main__' for value in values)


def audit(root: Path) -> dict:
    methods = []
    models = root / "motius" / "models"
    for directory in sorted(models.iterdir()):
        if not directory.is_dir() or directory.name.startswith("_"):
            continue
        files = sorted(directory.rglob("*.py"))
        if not files:
            continue
        findings = []
        for path in files:
            relative = path.relative_to(directory)
            filename = str(path.relative_to(root))
            reason = sorted(set(relative.parts[:-1]) & NON_NETWORK_PARTS)
            if reason or path.name in NON_NETWORK_FILES:
                findings.append({"path": filename, "line": 1, "kind": "non_network_file",
                                 "evidence": ", ".join(reason) or path.name})
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(path.read_text(encoding="utf-8"), filename=filename)
            except (SyntaxError, UnicodeDecodeError) as exc:
                findings.append({"path": filename, "line": getattr(exc, "lineno", 1),
                                 "kind": "parse_error", "evidence": str(exc)})
                continue
            for node in ast.walk(tree):
                if is_main_guard(node):
                    findings.append({"path": filename, "line": node.lineno,
                                     "kind": "application_entrypoint", "evidence": "__main__"})
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith(("motius.pipelines.", "motius.trainers.", "motius.datasets.")):
                        findings.append({"path": filename, "line": node.lineno,
                                         "kind": "non_network_dependency", "evidence": node.module})
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if (node.name.lower() in ORCHESTRATION_METHODS or node.name.startswith(("generate_", "sample_for_eval"))) and node.name not in ARCHITECTURE_GENERATORS:
                        findings.append({"path": filename, "line": node.lineno,
                                         "kind": "orchestration_in_models", "evidence": node.name})
                elif isinstance(node, ast.Call):
                    target = ast.unparse(node.func)
                    if target in {"subprocess.run", "subprocess.Popen", "os.system", "os.chdir"}:
                        findings.append({"path": filename, "line": node.lineno,
                                         "kind": "external_runtime", "evidence": target})
                    elif target in {"sys.path.insert", "sys.path.append", "sys.modules.setdefault"}:
                        findings.append({"path": filename, "line": node.lineno,
                                         "kind": "import_side_effect", "evidence": target})
                    elif (target == "clip.tokenize" or target in {"self.tokenizer", "self._tokenizer", "self.text_tokenizer", "self.llm_tokenizer", "self.sentence_emb_tokenizer"}
                          or target.endswith(".apply_chat_template")
                          or (target.endswith(".from_pretrained") and ("Tokenizer" in target or "tokenizer_class" in target))
                          or target.rsplit('.', 1)[-1] == "SentenceTransformer"):
                        findings.append({"path": filename, "line": node.lineno,
                                         "kind": "text_processing_in_models", "evidence": target})
        methods.append({"method": directory.name, "python_files": len(files),
                        "findings": findings, "source_parity": "unverified"})
    trainer_applications = []
    trainers = root / "motius" / "trainers"
    if trainers.exists():
        for directory in sorted(trainers.iterdir()):
            vendor = directory / "vendor"
            if directory.is_dir() and vendor.is_dir():
                trainer_applications.append({"method": directory.name,
                    "path": str(vendor.relative_to(root)),
                    "python_files": sum(1 for _ in vendor.rglob("*.py")),
                    "status": "pending_native_training_migration"})
    counts = Counter(f["kind"] for m in methods for f in m["findings"])
    return {"schema": "motius-method-architecture-v1", "methods": methods,
            "trainer_applications": trainer_applications,
            "summary": {"methods": len(methods),
                        "methods_with_findings": sum(bool(m["findings"]) for m in methods),
                        "python_files": sum(m["python_files"] for m in methods),
                        "findings": dict(sorted(counts.items()))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", action="append", default=[],
                        help="Limit the report/check to explicitly named methods")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = audit(args.root)
    if args.method:
        available = {entry["method"] for entry in report["methods"]}
        unknown = set(args.method) - available
        if unknown:
            parser.error(f"Unknown methods: {sorted(unknown)}")
        report["methods"] = [entry for entry in report["methods"] if entry["method"] in args.method]
        report["selection"] = args.method
        report["summary_all"] = report["summary"]
        counts = Counter(f["kind"] for entry in report["methods"] for f in entry["findings"])
        report["summary"] = {
            "methods": len(report["methods"]),
            "methods_with_findings": sum(bool(entry["findings"]) for entry in report["methods"]),
            "python_files": sum(entry["python_files"] for entry in report["methods"]),
            "findings": dict(sorted(counts.items())),
        }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    for method in report["methods"]:
        if method["findings"]:
            print(f"{method['method']}: {len(method['findings'])} findings")
    if args.check and any(m["findings"] for m in report["methods"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
