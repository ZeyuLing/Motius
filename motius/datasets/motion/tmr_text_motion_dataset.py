"""TMR text-motion dataset adapters owned by Motius."""

from __future__ import annotations

import codecs as cs
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, default_collate

try:
    import orjson
except ImportError:  # Optional fast path; stdlib JSON keeps evaluator-only envs portable.
    orjson = None

from motius.registry import DATASETS


def _load_json(path: Path) -> Any:
    with path.open("rb") as f:
        payload = f.read()
    if orjson is not None:
        return orjson.loads(payload)
    return json.loads(payload.decode("utf-8"))


def _read_split(path: Path, split: str) -> List[str]:
    split_file = path / "splits" / f"{split}.txt"
    with cs.open(split_file, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def _length_to_mask(length: List[int], device: Union[torch.device, str] = "cpu") -> Tensor:
    if isinstance(length, list):
        length = torch.tensor(length, device=device)
    max_len = max(length)
    return torch.arange(max_len, device=device).expand(len(length), max_len) < length.unsqueeze(1)


def _collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def _collate_x_dict(lst_x_dict: List[Dict[str, Any]], device: Optional[str] = None) -> Dict[str, Any]:
    x = _collate_tensor_with_padding([x_dict["x"] for x_dict in lst_x_dict])
    if device is not None:
        x = x.to(device)
    length = [x_dict["length"] for x_dict in lst_x_dict]
    mask = _length_to_mask(length, device=x.device)
    return {"x": x, "length": length, "mask": mask}


def collate_tmr_text_motion(lst_elements: List[Dict[str, Any]], device: Optional[str] = None) -> Dict[str, Any]:
    one_el = lst_elements[0]
    keys = one_el.keys()
    x_dict_keys = [key for key in keys if "x_dict" in key]
    other_keys = [key for key in keys if "x_dict" not in key]
    batch = {key: default_collate([x[key] for x in lst_elements]) for key in other_keys}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and device is not None:
            batch[key] = val.to(device)
    for key in x_dict_keys:
        batch[key] = _collate_x_dict([x[key] for x in lst_elements], device=device)
    return batch


class TMRNormalizer:
    def __init__(self, base_dir: str, eps: float = 1e-12, disable: bool = False):
        self.base_dir = Path(base_dir)
        self.mean_path = self.base_dir / "mean.pt"
        self.std_path = self.base_dir / "std.pt"
        self.eps = float(eps)
        self.disable = bool(disable)
        if not self.disable:
            self.mean = torch.load(self.mean_path)
            self.std = torch.load(self.std_path)

    def __call__(self, x: Tensor) -> Tensor:
        if self.disable:
            return x
        return (x - self.mean) / (self.std + self.eps)

    def inverse(self, x: Tensor) -> Tensor:
        if self.disable:
            return x
        return x * (self.std + self.eps) + self.mean


class TMRMotionLoader:
    def __init__(self, base_dir: str, fps: float, normalizer: Optional[TMRNormalizer] = None, nfeats: int = 38):
        self.base_dir = Path(base_dir)
        self.fps = float(fps)
        self.normalizer = normalizer
        self.nfeats = int(nfeats)
        self.motions: Dict[str, Tensor] = {}

    def __call__(self, path: str, start: float, end: float) -> Dict[str, Any]:
        begin = int(start * self.fps)
        finish = int(end * self.fps)
        if path not in self.motions:
            motion_path = self.base_dir / f"{path}.npy"
            motion = torch.from_numpy(np.load(motion_path)).to(torch.float32)
            if self.normalizer is not None:
                motion = self.normalizer(motion)
            self.motions[path] = motion
        motion = self.motions[path][begin:finish]
        return {"x": motion, "length": len(motion)}


class _PrecomputedEmbeddings:
    folder_name = ""

    def __init__(self, modelname: str, path: str, preload: bool = True):
        self.modelname = modelname
        self.embeddings_folder = Path(path) / self.folder_name
        self.cache: Dict[str, Any] = {}
        if preload:
            self.load_embeddings()
        else:
            self.embeddings_index: Dict[str, int] = {}

    def __contains__(self, text: str) -> bool:
        return text in self.embeddings_index

    def __call__(self, texts: Union[str, Iterable[str]]) -> Any:
        squeeze = False
        if isinstance(texts, str):
            texts = [texts]
            squeeze = True
        out = [self.get_embedding(text) for text in texts]
        return out[0] if squeeze else out


class TMRTokenEmbeddings(_PrecomputedEmbeddings):
    folder_name = "token_embeddings"

    def load_embeddings(self) -> None:
        npy = self.embeddings_folder / f"{self.modelname}.npy"
        self.embeddings_big = torch.from_numpy(np.load(npy, mmap_mode="c")).to(torch.float32)
        self.embeddings_slice = np.load(self.embeddings_folder / f"{self.modelname}_slice.npy")
        self.embeddings_index = _load_json(self.embeddings_folder / f"{self.modelname}_index.json")

    def get_embedding(self, text: str) -> Dict[str, Any]:
        index = self.embeddings_index[text]
        begin, end = self.embeddings_slice[index]
        embedding = self.embeddings_big[begin:end]
        return {"x": embedding, "length": len(embedding)}


class TMRSentenceEmbeddings(_PrecomputedEmbeddings):
    folder_name = "sent_embeddings"

    def load_embeddings(self) -> None:
        npy = self.embeddings_folder / f"{self.modelname}.npy"
        self.embeddings = torch.from_numpy(np.load(npy, mmap_mode="c")).to(torch.float32)
        self.embeddings_index = _load_json(self.embeddings_folder / f"{self.modelname}_index.json")

    def get_embedding(self, text: str) -> Tensor:
        return self.embeddings[self.embeddings_index[text]]


@DATASETS.register_module()
class TMRTextMotionDataset(Dataset):
    """Materialized TMR dataset for Motius configs.

    This adapter keeps the on-disk TMR format used by the official
    implementation (``annotations.json``, ``splits/*.txt``, ``motions/*.npy``,
    precomputed text embeddings, and ``stats``), but exposes it through the
    normal Motius dataset registry without depending on a vendored source tree.
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        fps: float = 30.0,
        nfeats: int = 38,
        motion_dir: Optional[str] = None,
        stats_dir: Optional[str] = None,
        token_modelname: str = "distilbert-base-uncased",
        sentence_modelname: str = "sentence-transformers/all-mpnet-base-v2",
        min_seconds: float = 0.5,
        max_seconds: float = 120.0,
        preload: bool = False,
        tiny: bool = False,
        disable_normalizer: bool = False,
    ) -> None:
        super().__init__()
        dataset_path = Path(dataset_dir)
        motion_path = Path(motion_dir) if motion_dir else dataset_path / "motions"
        stats_path = Path(stats_dir) if stats_dir else dataset_path / "stats"
        if tiny:
            split = split + "_tiny"

        normalizer = TMRNormalizer(
            base_dir=str(stats_path),
            disable=bool(disable_normalizer),
        )
        self.motion_loader = TMRMotionLoader(
            base_dir=str(motion_path),
            fps=float(fps),
            normalizer=normalizer,
            nfeats=int(nfeats),
        )
        self.text_to_token_emb = TMRTokenEmbeddings(
            modelname=token_modelname,
            path=str(dataset_path),
            preload=True,
        )
        self.text_to_sent_emb = TMRSentenceEmbeddings(
            modelname=sentence_modelname,
            path=str(dataset_path),
            preload=True,
        )

        self.path = dataset_path
        self.collate_fn = collate_tmr_text_motion
        self.split = split
        self.keyids = _read_split(dataset_path, split)
        self.min_seconds = float(min_seconds)
        self.max_seconds = float(max_seconds)
        self.annotations = _load_json(dataset_path / "annotations.json")
        if "test" not in split:
            self.annotations = self.filter_annotations(self.annotations)
        self.is_training = split == "train"
        self.keyids = [keyid for keyid in self.keyids if keyid in self.annotations]
        self.nfeats = self.motion_loader.nfeats
        if preload:
            for _ in self:
                continue

    def __len__(self) -> int:
        return len(self.keyids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.load_keyid(self.keyids[index])

    def load_keyid(self, keyid: str) -> Dict[str, Any]:
        annotations = self.annotations[keyid]
        index = 0
        if self.is_training:
            index = np.random.randint(len(annotations["annotations"]))
        annotation = annotations["annotations"][index]
        text = annotation["text"]
        return {
            "motion_x_dict": self.motion_loader(
                path=annotations["path"],
                start=annotation["start"],
                end=annotation["end"],
            ),
            "text_x_dict": self.text_to_token_emb(text),
            "text": text,
            "keyid": keyid,
            "sent_emb": self.text_to_sent_emb(text),
        }

    def filter_annotations(self, annotations: Dict[str, Any]) -> Dict[str, Any]:
        filtered = {}
        for key, val in annotations.items():
            item = dict(val)
            annots = item.pop("annotations")
            kept = []
            for annot in annots:
                duration = annot["end"] - annot["start"]
                if self.max_seconds >= duration >= self.min_seconds:
                    kept.append(annot)
            if kept:
                item["annotations"] = kept
                filtered[key] = item
        return filtered
