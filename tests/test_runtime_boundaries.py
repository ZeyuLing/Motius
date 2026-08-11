from __future__ import annotations

import subprocess


def test_method_runtime_has_no_legacy_repository_imports():
    result = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "-i",
            "-E",
            r"^\s*(from|import)\s+.*(ref_repo|hf_trainer)",
            "--",
            "motius/models/**/*.py",
            "motius/pipelines/**/*.py",
            "motius/trainers/**/*.py",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stdout
