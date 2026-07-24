"""Subprocess-isolated monocular capture pipeline for official GEM-SMPL."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Sequence

from motius.models.gem_smpl import GemSmplBundle
from motius.models.gem_smpl.runtime import (
    build_demo_command,
    expected_output_path,
    runtime_python,
    verify_checkpoint,
    verify_runtime_checkout,
)
from motius.pipelines.base_pipeline import BasePipeline
from motius.pipelines.gem_smpl.parser import parse_gem_smpl_file
from motius.registry import PIPELINES


@PIPELINES.register_module()
class GemSmplPipeline(BasePipeline):
    """Run the pinned noncommercial GEM-SMPL runtime out of process."""

    BUNDLE_CLS = "motius.models.gem_smpl.GemSmplBundle"

    def __init__(self, bundle: GemSmplBundle):
        super().__init__(bundle)

    def build_command(
        self,
        video: str | Path,
        output_root: str | Path,
        *,
        static_camera: bool = False,
        extra_args: Sequence[str] = (),
    ) -> tuple[str, ...]:
        return build_demo_command(
            runtime_root=self.bundle.runtime_root,
            video=video,
            output_root=output_root,
            checkpoint=self.bundle.checkpoint,
            python_executable=self.bundle.python_executable,
            static_camera=static_camera,
            extra_args=extra_args,
        )

    def build_export_command(
        self,
        official_output: str | Path,
        numeric_output: str | Path,
    ) -> tuple[str, ...]:
        script = Path(__file__).with_name("export_native.py").resolve()
        return (
            str(
                runtime_python(
                    self.bundle.runtime_root,
                    self.bundle.python_executable,
                )
            ),
            str(script),
            "--input",
            str(Path(official_output).resolve()),
            "--output",
            str(Path(numeric_output).resolve()),
            "--device",
            self.bundle.export_device,
        )

    def run(
        self,
        video: str | Path,
        output_root: str | Path,
        *,
        original_fps: float,
        static_camera: bool = False,
        extra_args: Sequence[str] = (),
        timeout: Optional[float] = None,
    ):
        """Execute real inference, official SMPL24 FK, then parse numeric output."""

        verify_runtime_checkout(self.bundle.runtime_root)
        verify_checkpoint(self.bundle.checkpoint)
        output_root = Path(output_root).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            self.build_command(
                video,
                output_root,
                static_camera=static_camera,
                extra_args=extra_args,
            ),
            cwd=self.bundle.runtime_root,
            check=True,
            timeout=timeout,
        )
        official_output = expected_output_path(video, output_root)
        if not official_output.is_file():
            raise FileNotFoundError(
                f"Official GEM-SMPL demo did not produce {official_output}."
            )
        numeric_output = official_output.with_name("motius_smpl_native.npz")
        subprocess.run(
            self.build_export_command(official_output, numeric_output),
            cwd=self.bundle.runtime_root,
            check=True,
            timeout=timeout,
        )
        return parse_gem_smpl_file(numeric_output, original_fps=original_fps)


GemSmplMonocularPipeline = GemSmplPipeline


__all__ = ["GemSmplMonocularPipeline", "GemSmplPipeline"]
