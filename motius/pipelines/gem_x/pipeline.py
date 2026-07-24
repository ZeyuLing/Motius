"""Subprocess-isolated monocular capture pipeline for official GEM-X."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from motius.models.gem_x import GemXBundle
from motius.models.gem_x.runtime import (
    build_demo_command,
    expected_output_path,
    runtime_python,
    verify_checkpoint,
    verify_runtime_checkout,
)
from motius.pipelines.base_pipeline import BasePipeline
from motius.pipelines.gem_x.parser import parse_gem_x_file
from motius.registry import PIPELINES


@PIPELINES.register_module()
class GemXPipeline(BasePipeline):
    """Run native SOMA-77 inference without importing it into Motius."""

    BUNDLE_CLS = "motius.models.gem_x.GemXBundle"

    def __init__(self, bundle: GemXBundle):
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
            "--soma-assets",
            self.bundle.soma_assets,
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
        """Execute real inference and official SOMA FK, then parse native output."""

        verify_runtime_checkout(self.bundle.runtime_root)
        verify_checkpoint(self.bundle.checkpoint)
        output_root = Path(output_root).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        process_env = os.environ.copy()
        process_env["MOTIUS_GEM_X_SKIP_RENDER"] = "1"
        subprocess.run(
            self.build_command(
                video,
                output_root,
                static_camera=static_camera,
                extra_args=extra_args,
            ),
            cwd=self.bundle.runtime_root,
            env=process_env,
            check=True,
            timeout=timeout,
        )
        official_output = expected_output_path(video, output_root)
        # docs/DEMO.md at this revision still mentions the older preprocess path.
        legacy_output = official_output.parent / "preprocess" / official_output.name
        if not official_output.is_file() and legacy_output.is_file():
            official_output = legacy_output
        if not official_output.is_file():
            raise FileNotFoundError(f"Official GEM-X demo did not produce {official_output}.")
        numeric_output = official_output.with_name("motius_soma77_native.npz")
        subprocess.run(
            self.build_export_command(official_output, numeric_output),
            cwd=self.bundle.runtime_root,
            check=True,
            timeout=timeout,
        )
        return parse_gem_x_file(numeric_output, original_fps=original_fps)


GemXMonocularPipeline = GemXPipeline


__all__ = ["GemXMonocularPipeline", "GemXPipeline"]
