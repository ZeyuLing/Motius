"""Formatting transforms for dataset pipelines."""

from __future__ import annotations

from typing import Dict, Iterable

from hftrainer.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RenameKeys:
    """Rename keys in the sample dict."""

    def __init__(self, mapping: Dict[str, str], pop_source: bool = True):
        self.mapping = dict(mapping)
        self.pop_source = pop_source

    def __call__(self, results: dict) -> dict:
        for source, target in self.mapping.items():
            if source not in results:
                continue
            value = results.pop(source) if self.pop_source else results[source]
            results[target] = value
        return results


@TRANSFORMS.register_module()
class PackMetaKeys:
    """Collect selected keys under ``metas`` while keeping task inputs flat."""

    def __init__(
        self,
        meta_keys: Iterable[str],
        output_key: str = 'metas',
    ):
        self.meta_keys = list(meta_keys)
        self.output_key = output_key

    def __call__(self, results: dict) -> dict:
        metas = {}
        for key in self.meta_keys:
            if key in results:
                metas[key] = results[key]
        if metas:
            results[self.output_key] = metas
        return results
