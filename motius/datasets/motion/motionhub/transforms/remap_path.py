"""Remap motion file paths between different motion representations.

Used to convert annotation paths (e.g., pointing to .npz files in motions/)
to pre-processed npy paths (e.g., in motions_o6dp_v1205/).
"""

import os
from typing import Dict

from mmcv import BaseTransform

from motius.registry import TRANSFORMS


@TRANSFORMS.register_module(force=True)
class RemapMotionPathToO6dp(BaseTransform):
    """Remap motion_path from source directory/extension to destination.

    Transforms the motion file path in results dict by replacing the
    directory component and file extension. Useful for loading
    pre-processed o6dp npy files instead of raw npz files.

    Example:
        If motion_path = '.../motions/HumanML3D-HumanEva/foo.npz'
        with src_dir='motions', dst_dir='motions_o6dp_v1205',
        src_ext='.npz', dst_ext='.npy',
        result = '.../motions_o6dp_v1205/HumanML3D-HumanEva/foo.npy'

    Parameters
    ----------
    key : str
        The key whose path will be remapped (default: 'motion').
    src_dir : str
        Directory name to replace in the path.
    dst_dir : str
        Replacement directory name.
    src_ext : str
        File extension to replace.
    dst_ext : str
        Replacement file extension.
    """

    def __init__(
        self,
        key: str = 'motion',
        src_dir: str = 'motions',
        dst_dir: str = 'motions_o6dp_v1205',
        src_ext: str = '.npz',
        dst_ext: str = '.npy',
    ):
        super().__init__()
        self.key = key
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.src_ext = src_ext
        self.dst_ext = dst_ext

    def _remap_path(self, path: str) -> str:
        # Replace directory component: /xxx/motions/yyy -> /xxx/motions_o6dp_v1205/yyy
        path = path.replace(f'/{self.src_dir}/', f'/{self.dst_dir}/')
        # Replace extension
        if path.endswith(self.src_ext):
            path = path[:-len(self.src_ext)] + self.dst_ext
        return path

    def transform(self, results: Dict) -> Dict:
        path_key = f'{self.key}_path'
        if path_key not in results:
            return results

        path_or_list = results[path_key]
        if isinstance(path_or_list, str):
            new_path = self._remap_path(path_or_list)
            if not os.path.exists(new_path):
                raise FileNotFoundError(
                    f"RemapMotionPathToO6dp: remapped path does not exist: {new_path} "
                    f"(original: {path_or_list})"
                )
            results[path_key] = new_path
        elif isinstance(path_or_list, (list, tuple)):
            new_paths = []
            for p in path_or_list:
                new_p = self._remap_path(str(p))
                if not os.path.exists(new_p):
                    raise FileNotFoundError(
                        f"RemapMotionPathToO6dp: remapped path does not exist: {new_p} "
                        f"(original: {p})"
                    )
                new_paths.append(new_p)
            results[path_key] = new_paths

        return results
