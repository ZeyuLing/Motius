"""Shared data contract and text metrics for motion-to-text evaluation."""

from __future__ import annotations

import json
import math
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


HUMANML3D_FPS = 20.0
HUMANML3D_MIN_FRAMES = 40
HUMANML3D_MAX_FRAMES = 196
HUMANML3D_MAX_SOURCE_FRAMES = 200
HUMANML3D_REFERENCE_COUNT = 3
HUMANML3D_M2T_PROTOCOL_VERSION = "humanml3d_tm2t_v2"


@dataclass(frozen=True)
class HumanML3DCaption:
    """One line from a HumanML3D ``texts/<id>.txt`` annotation."""

    caption: str
    tokens: Tuple[str, ...]
    from_tag: float
    to_tag: float

    @property
    def is_full_motion(self) -> bool:
        return self.from_tag == 0.0 and self.to_tag == 0.0

    @property
    def token_text(self) -> str:
        return " ".join(token.rsplit("/", 1)[0] for token in self.tokens)


@dataclass(frozen=True)
class HumanML3DM2TSample:
    """A lazy HumanML3D motion-caption sample used by every M2T baseline."""

    sample_id: str
    source_id: str
    motion_path: Path
    start_frame: int
    end_frame: int
    captions: Tuple[HumanML3DCaption, ...]

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def references(self) -> Tuple[str, ...]:
        return tuple(item.caption for item in self.captions)

    @property
    def token_references(self) -> Tuple[str, ...]:
        return tuple(item.token_text for item in self.captions)

    def load_motion(self) -> np.ndarray:
        motion = np.load(self.motion_path, mmap_mode="r")
        clip = np.array(
            motion[self.start_frame : self.end_frame], dtype=np.float32, copy=True
        )
        if clip.ndim != 2 or clip.shape[1] != 263:
            raise ValueError(
                f"Expected HumanML3D-263 motion for {self.sample_id}, got {clip.shape}."
            )
        return clip

    def to_prediction_record(self, prediction: str) -> dict:
        """Serialize the model-independent record consumed by the evaluator."""

        return {
            "id": self.sample_id,
            "source_id": self.source_id,
            "prediction": str(prediction).strip(),
            "references": list(official_reference_count(self.references)),
            "token_references": list(official_reference_count(self.token_references)),
            "length": self.length,
            "motion_path": str(self.motion_path),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }


def _finite_tag(value: str) -> float:
    tag = float(value)
    return 0.0 if not math.isfinite(tag) else tag


def read_humanml3d_captions(path: Union[str, Path]) -> List[HumanML3DCaption]:
    """Parse a HumanML3D caption file without discarding temporal annotations."""

    captions: List[HumanML3DCaption] = []
    path = Path(path)
    if not path.exists():
        return captions
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw_line.strip().split("#")
        if len(parts) < 4:
            continue
        try:
            from_tag = _finite_tag(parts[2])
            to_tag = _finite_tag(parts[3])
        except ValueError:
            continue
        caption = parts[0].strip()
        tokens = tuple(token for token in parts[1].split() if token)
        if caption and tokens:
            captions.append(HumanML3DCaption(caption, tokens, from_tag, to_tag))
    return captions


def official_reference_count(
    references: Sequence[str], count: int = HUMANML3D_REFERENCE_COUNT
) -> Tuple[str, ...]:
    """Apply TM2T's fixed three-reference policy.

    TM2T keeps the first three descriptions, duplicates a two-reference sample's
    first description once, and repeats a single reference three times.
    """

    values = tuple(str(value).strip() for value in references if str(value).strip())
    if not values:
        raise ValueError("At least one non-empty M2T reference is required.")
    if len(values) >= count:
        return values[:count]
    return values + tuple(values[index % len(values)] for index in range(count - len(values)))


def load_humanml3d_m2t_samples(
    data_root: Union[str, Path],
    split_file: Union[str, Path] = "test.txt",
    *,
    include_subclips: bool = True,
    min_frames: int = HUMANML3D_MIN_FRAMES,
    max_frames: int = HUMANML3D_MAX_FRAMES,
    max_source_frames: int = HUMANML3D_MAX_SOURCE_FRAMES,
    max_samples: Optional[int] = None,
    io_workers: int = 32,
) -> List[HumanML3DM2TSample]:
    """Build the official TM2T test population from a HumanML3D layout.

    Full-motion captions are grouped as multiple references. A caption with
    temporal tags creates an independently scored subclip. Population order,
    duplicate temporal tags, and dictionary overwrite behavior intentionally
    match TM2T's released ``Motion2TextEvalDataset``. The released loader
    accepts clips in ``[40, 200)`` and truncates evaluator inputs to 196 frames.
    """

    data_root = Path(data_root)
    split_path = Path(split_file)
    if not split_path.is_absolute():
        split_path = data_root / split_path
    if not split_path.exists():
        raise FileNotFoundError(f"HumanML3D split file not found: {split_path}")

    motion_dir = data_root / "new_joint_vecs"
    text_dir = data_root / "texts"
    source_ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    def load_source(
        source_id: str,
    ) -> List[Tuple[str, HumanML3DM2TSample]]:
        motion_path = motion_dir / f"{source_id}.npy"
        if not motion_path.exists():
            return []
        motion = np.load(motion_path, mmap_mode="r")
        if motion.ndim != 2 or motion.shape[1] != 263:
            return []
        motion_length = int(motion.shape[0])
        if motion_length < min_frames or motion_length >= max_source_frames:
            return []

        captions = read_humanml3d_captions(text_dir / f"{source_id}.txt")
        full = tuple(item for item in captions if item.is_full_motion)
        source_samples: List[Tuple[str, HumanML3DM2TSample]] = []
        if include_subclips:
            for item in captions:
                if item.is_full_motion:
                    continue
                start = max(0, int(item.from_tag * HUMANML3D_FPS))
                end = min(motion_length, int(item.to_tag * HUMANML3D_FPS))
                if end - start < min_frames or end - start >= max_source_frames:
                    continue
                sample_id = f"{source_id}_{item.from_tag:.6f}_{item.to_tag:.6f}"
                source_samples.append(
                    (
                        sample_id,
                        HumanML3DM2TSample(
                            sample_id=sample_id,
                            source_id=source_id,
                            motion_path=motion_path,
                            start_frame=start,
                            end_frame=min(end, start + max_frames),
                            captions=(item,),
                        ),
                    )
                )
        if full:
            source_samples.append(
                (
                    source_id,
                    HumanML3DM2TSample(
                        sample_id=source_id,
                        source_id=source_id,
                        motion_path=motion_path,
                        start_frame=0,
                        end_frame=min(motion_length, max_frames),
                        captions=full,
                    ),
                )
            )
        return source_samples

    # The official dataset appends every key to name_list, overwrites duplicate
    # keys in data_dict, and reports len(data_dict). Reproduce that behavior so
    # published TM2T numbers and Motius numbers use the same test population.
    ordered_ids: List[str] = []
    samples_by_id = {}
    workers = max(1, int(io_workers))
    batch_size = max(1, workers * 8)
    if max_samples is not None:
        batch_size = min(batch_size, max(workers, max_samples * 2))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(source_ids), batch_size):
            groups = executor.map(load_source, source_ids[start : start + batch_size])
            for group in groups:
                for sample_id, sample in group:
                    ordered_ids.append(sample_id)
                    samples_by_id[sample_id] = sample
    samples = [
        samples_by_id[sample_id]
        for sample_id in ordered_ids[: len(samples_by_id)]
    ]
    return samples if max_samples is None else samples[:max_samples]


def write_humanml3d_m2t_manifest(
    samples: Sequence[HumanML3DM2TSample],
    path: Union[str, Path],
    *,
    data_root: Union[str, Path],
    split_file: str = "test.txt",
) -> Path:
    """Persist one shared, relocatable M2T evaluation population."""

    root = Path(data_root).expanduser().resolve()
    rows = []
    for sample in samples:
        motion_path = sample.motion_path.expanduser().resolve()
        try:
            relative_motion_path = motion_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Motion path is outside data_root: {motion_path}") from exc
        rows.append(
            {
                "sample_id": sample.sample_id,
                "source_id": sample.source_id,
                "motion_path": str(relative_motion_path),
                "start_frame": sample.start_frame,
                "end_frame": sample.end_frame,
                "captions": [
                    {
                        "caption": caption.caption,
                        "tokens": list(caption.tokens),
                        "from_tag": caption.from_tag,
                        "to_tag": caption.to_tag,
                    }
                    for caption in sample.captions
                ],
            }
        )
    payload = {
        "protocol": HUMANML3D_M2T_PROTOCOL_VERSION,
        "dataset": "HumanML3D",
        "split_file": str(split_file),
        "data_root": str(root),
        "frame_policy": "accept_[40,200)_truncate_196",
        "reference_policy": "tm2t_first3_repeat_to3",
        "num_samples": len(rows),
        "samples": rows,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def load_humanml3d_m2t_manifest(
    path: Union[str, Path],
    *,
    data_root: Optional[Union[str, Path]] = None,
) -> List[HumanML3DM2TSample]:
    """Load a shared M2T population, optionally relocating its dataset root."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("protocol") != HUMANML3D_M2T_PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported M2T protocol manifest: {payload.get('protocol')!r}"
        )
    root = Path(data_root or payload["data_root"]).expanduser().resolve()
    samples = []
    for row in payload.get("samples", []):
        captions = tuple(
            HumanML3DCaption(
                caption=str(item["caption"]),
                tokens=tuple(item["tokens"]),
                from_tag=float(item["from_tag"]),
                to_tag=float(item["to_tag"]),
            )
            for item in row["captions"]
        )
        samples.append(
            HumanML3DM2TSample(
                sample_id=str(row["sample_id"]),
                source_id=str(row["source_id"]),
                motion_path=root / row["motion_path"],
                start_frame=int(row["start_frame"]),
                end_frame=int(row["end_frame"]),
                captions=captions,
            )
        )
    if len(samples) != int(payload.get("num_samples", len(samples))):
        raise ValueError(f"Truncated M2T protocol manifest: {manifest_path}")
    return samples


class HumanML3DTextNormalizer:
    """TM2T-compatible lemmatization for generated descriptions."""

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        try:
            import spacy

            self.nlp = spacy.load(spacy_model)
        except Exception as exc:
            raise RuntimeError(
                "M2T text normalization requires spaCy's en_core_web_sm model. "
                "Install it with `python -m spacy download en_core_web_sm`."
            ) from exc

    @staticmethod
    def _normalize_doc(doc) -> str:
        words = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if token.pos_ in {"NOUN", "VERB"} and word != "left":
                word = token.lemma_
            words.append(word.lower())
        return " ".join(words)

    def normalize_many(self, sentences: Sequence[str]) -> List[str]:
        values = [str(sentence).replace("-", "") for sentence in sentences]
        return [
            self._normalize_doc(doc)
            for doc in self.nlp.pipe(values, batch_size=512)
        ]

    def __call__(self, sentence: str) -> str:
        return self.normalize_many([sentence])[0]


def compute_coco_caption_metrics(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
) -> dict:
    """Compute TM2T's BLEU, ROUGE-L, and CIDEr metric family.

    ``pycocoevalcap`` contains the same COCO scorers used by ``nlg-eval`` in
    the official TM2T release, while avoiding unrelated Java-only metrics.
    """

    if len(predictions) != len(references):
        raise ValueError("Predictions and reference groups must have equal length.")
    if not predictions:
        raise ValueError("At least one M2T prediction is required.")
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError as exc:
        raise RuntimeError(
            "M2T linguistic metrics require the `m2t` optional dependencies: "
            "install Motius with `pip install -e '.[m2t]'`."
        ) from exc

    refs = {
        index: list(official_reference_count(group))
        for index, group in enumerate(references)
    }
    hyps = {index: [str(value).strip()] for index, value in enumerate(predictions)}
    bleu, _ = Bleu(4).compute_score(refs, hyps)
    rouge, _ = Rouge().compute_score(refs, hyps)
    cider, _ = Cider().compute_score(refs, hyps)
    return {
        "Bleu_1": float(bleu[0]),
        "Bleu_2": float(bleu[1]),
        "Bleu_3": float(bleu[2]),
        "Bleu_4": float(bleu[3]),
        "ROUGE_L": float(rouge),
        "CIDEr": float(cider),
    }


def _bert_score_f1_baseline(
    *,
    language: str,
    model_type: Optional[str],
) -> Tuple[float, str, int]:
    """Resolve the baseline used by ``bert-score`` for F1 rescaling."""

    import bert_score
    from bert_score.utils import lang2model, model2layers

    resolved_model = model_type or lang2model[language]
    layer = int(model2layers[resolved_model])
    baseline_path = (
        Path(bert_score.__file__).resolve().parent
        / "rescale_baseline"
        / language
        / f"{resolved_model}.tsv"
    )
    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"BERTScore baseline is unavailable for {resolved_model!r}: {baseline_path}"
        )
    with baseline_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["LAYER"]) == layer:
                return float(row["F"]), resolved_model, layer
    raise ValueError(
        f"BERTScore baseline {baseline_path} has no row for layer {layer}."
    )


def compute_bert_scores(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    *,
    device: str = "cuda",
    model_type: Optional[str] = None,
    batch_size: int = 64,
) -> dict:
    """Compute raw and paper-compatible multi-reference BERTScore F1.

    BERTScore's raw value is a contextual-embedding cosine score and is usually
    close to 0.9 for English captions. TM2T reports the baseline-rescaled value,
    which is commonly around 0.3. Returning both prevents those two scales from
    being compared as though they were different model results.
    """

    try:
        from bert_score import score
    except ImportError as exc:
        raise RuntimeError(
            "BERTScore requires the `m2t` optional dependencies: "
            "install Motius with `pip install -e '.[m2t]'`."
        ) from exc
    kwargs = {
        "lang": "en",
        "rescale_with_baseline": False,
        "idf": True,
        "device": device,
        "batch_size": int(batch_size),
        "verbose": False,
    }
    if model_type:
        kwargs["model_type"] = model_type
    _, _, raw_f1 = score(
        list(map(str, predictions)),
        [list(official_reference_count(group)) for group in references],
        **kwargs,
    )
    baseline, resolved_model, layer = _bert_score_f1_baseline(
        language="en", model_type=model_type
    )
    rescaled_f1 = (raw_f1 - baseline) / (1.0 - baseline)
    return {
        "raw": float(raw_f1.mean().item()),
        "rescaled": float(rescaled_f1.mean().item()),
        "baseline": float(baseline),
        "model_type": resolved_model,
        "layer": layer,
    }


def compute_bert_score(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    *,
    device: str = "cuda",
    model_type: Optional[str] = None,
    batch_size: int = 64,
) -> float:
    """Return TM2T's baseline-rescaled BERTScore F1 for compatibility."""

    return compute_bert_scores(
        predictions,
        references,
        device=device,
        model_type=model_type,
        batch_size=batch_size,
    )["rescaled"]


def write_prediction_records(
    output_dir: Union[str, Path],
    rows: Iterable[Tuple[HumanML3DM2TSample, str]],
) -> int:
    """Write resumable per-sample prediction JSON files."""

    prediction_dir = Path(output_dir) / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for sample, prediction in rows:
        path = prediction_dir / f"{sample.sample_id}.json"
        path.write_text(
            json.dumps(sample.to_prediction_record(prediction), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        count += 1
    return count


__all__ = [
    "HUMANML3D_MAX_FRAMES",
    "HUMANML3D_MAX_SOURCE_FRAMES",
    "HUMANML3D_M2T_PROTOCOL_VERSION",
    "HUMANML3D_MIN_FRAMES",
    "HUMANML3D_REFERENCE_COUNT",
    "HumanML3DCaption",
    "HumanML3DM2TSample",
    "HumanML3DTextNormalizer",
    "compute_bert_score",
    "compute_bert_scores",
    "compute_coco_caption_metrics",
    "load_humanml3d_m2t_samples",
    "load_humanml3d_m2t_manifest",
    "official_reference_count",
    "read_humanml3d_captions",
    "write_prediction_records",
    "write_humanml3d_m2t_manifest",
]
