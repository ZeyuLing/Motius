"""HumanML3D motion-to-text evaluator."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from motius.evaluation.m2t import (
    HumanML3DTextNormalizer,
    compute_bert_scores,
    compute_coco_caption_metrics,
    load_humanml3d_m2t_manifest,
    official_reference_count,
)
from motius.evaluation.metrics import r_precision
from motius.registry import EVALUATORS


@EVALUATORS.register_module()
class HumanMLM2TEvaluator:
    """Score model-independent HumanML3D M2T prediction records.

    Linguistic metrics follow the original TM2T protocol. Passing a semantic
    evaluator with ``encode_texts`` and ``encode_motions`` additionally enables
    Matching Distance and R-Precision with candidate batches of 32.
    """

    def __init__(
        self,
        *,
        semantic_evaluator=None,
        chunk_size: int = 32,
        n_repeats: int = 1,
        seed: int = 42,
        normalize_predictions: bool = True,
        spacy_model: str = "en_core_web_sm",
        compute_bertscore: bool = True,
        bert_device: str = "cuda",
        bert_model_type: Optional[str] = None,
        bert_batch_size: int = 64,
        language_reference_mode: str = "token",
        io_workers: int = 32,
    ) -> None:
        self.semantic_evaluator = semantic_evaluator
        self.chunk_size = int(chunk_size)
        self.n_repeats = int(n_repeats)
        self.seed = int(seed)
        self.normalize_predictions = bool(normalize_predictions)
        self.spacy_model = str(spacy_model)
        self.compute_bertscore = bool(compute_bertscore)
        self.bert_device = str(bert_device)
        self.bert_model_type = bert_model_type
        self.bert_batch_size = max(1, int(bert_batch_size))
        if language_reference_mode not in {"token", "raw"}:
            raise ValueError(
                "language_reference_mode must be either 'token' or 'raw', got "
                f"{language_reference_mode!r}."
            )
        self.language_reference_mode = language_reference_mode
        self.io_workers = max(1, int(io_workers))
        self._normalizer = None

    @staticmethod
    def load_prediction_records(
        prediction_dir: Union[str, Path],
        max_samples: Optional[int] = None,
        *,
        protocol_manifest: Optional[Union[str, Path]] = None,
        data_root: Optional[Union[str, Path]] = None,
        io_workers: int = 32,
    ) -> List[Dict]:
        path = Path(prediction_dir)
        if (path / "predictions").is_dir():
            path = path / "predictions"
        records: List[Dict] = []
        if protocol_manifest:
            samples = load_humanml3d_m2t_manifest(
                protocol_manifest, data_root=data_root
            )
            if max_samples is not None:
                samples = samples[:max_samples]
            unique_ids = list(dict.fromkeys(sample.sample_id for sample in samples))

            def read_prediction(sample_id: str):
                record_path = path / f"{sample_id}.json"
                if not record_path.exists():
                    return sample_id, None, record_path
                prediction = str(
                    json.loads(record_path.read_text(encoding="utf-8")).get(
                        "prediction", ""
                    )
                ).strip()
                return sample_id, prediction, record_path

            with ThreadPoolExecutor(max_workers=max(1, int(io_workers))) as executor:
                loaded = dict(
                    (sample_id, (prediction, record_path))
                    for sample_id, prediction, record_path in executor.map(
                        read_prediction, unique_ids
                    )
                )
            missing = [sample_id for sample_id in unique_ids if loaded[sample_id][0] is None]
            if missing:
                preview = ", ".join(missing[:5])
                raise FileNotFoundError(
                    f"Missing {len(missing)} M2T predictions required by the "
                    f"protocol manifest; first: {preview}"
                )
            for sample in samples:
                prediction, record_path = loaded[sample.sample_id]
                record = sample.to_prediction_record(prediction)
                record["metric_references"] = list(
                    official_reference_count(sample.token_references)
                )
                record["_record_path"] = str(record_path)
                records.append(record)
            return records

        record_paths = sorted(path.glob("*.json"))
        if max_samples is not None:
            record_paths = record_paths[:max_samples]

        def read_record(record_path: Path):
            record = json.loads(record_path.read_text(encoding="utf-8"))
            references = record.get("token_references") or record.get("references") or []
            if not references:
                return None
            record["prediction"] = str(record.get("prediction", "")).strip()
            record["metric_references"] = list(official_reference_count(references))
            record["_record_path"] = str(record_path)
            return record

        with ThreadPoolExecutor(max_workers=max(1, int(io_workers))) as executor:
            records.extend(
                record for record in executor.map(read_record, record_paths) if record
            )
        return records

    def _prediction_texts(self, records: Sequence[Dict]) -> List[str]:
        predictions = [str(record["prediction"]).strip() for record in records]
        if not self.normalize_predictions:
            return predictions
        if self._normalizer is None:
            self._normalizer = HumanML3DTextNormalizer(self.spacy_model)
        return self._normalizer.normalize_many(predictions)

    def _language_references(self, record: Dict) -> List[str]:
        if self.language_reference_mode == "raw":
            references = record.get("references") or record["metric_references"]
        else:
            references = record.get("token_references") or record["metric_references"]
        return list(official_reference_count(references))

    @staticmethod
    def build_gt_records(
        protocol_manifest: Union[str, Path],
        *,
        data_root: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = None,
    ) -> List[Dict]:
        """Use each protocol sample's first reference as the GT baseline."""

        samples = load_humanml3d_m2t_manifest(
            protocol_manifest, data_root=data_root
        )
        if max_samples is not None:
            samples = samples[:max_samples]
        records = []
        for sample in samples:
            record = sample.to_prediction_record(sample.references[0])
            record["metric_references"] = list(
                official_reference_count(sample.token_references)
            )
            records.append(record)
        return records

    @staticmethod
    def _load_motion(record: Dict) -> np.ndarray:
        path = Path(record["motion_path"])
        motion = np.load(path, mmap_mode="r")
        start = int(record.get("start_frame", 0))
        end = int(record.get("end_frame", record.get("length", len(motion))))
        clip = np.array(motion[start:end], dtype=np.float32, copy=True)
        if clip.ndim != 2 or clip.shape[1] != 263:
            raise ValueError(f"Expected HumanML3D-263 motion in {path}, got {clip.shape}.")
        return clip

    def _semantic_metrics(
        self,
        records: Sequence[Dict],
        predictions: Sequence[str],
    ) -> dict:
        evaluator = self.semantic_evaluator
        count = len(records)
        motions = [self._load_motion(record) for record in records]
        generated = evaluator.encode_texts(predictions)
        reference_texts = [
            text for record in records for text in record["metric_references"]
        ]
        references = evaluator.encode_texts(reference_texts).reshape(count, 3, -1)
        motion_embeddings = evaluator.encode_motions(motions)

        chunk = self.chunk_size
        used = count // chunk * chunk
        if used == 0:
            raise ValueError(
                f"M2T R-Precision requires at least one full candidate batch of {chunk}; "
                f"got {count} records."
            )
        rng = np.random.default_rng(self.seed)
        pred_precision, gt_precision = [], []
        pred_matching, gt_matching = [], []
        for _ in range(self.n_repeats):
            order = rng.permutation(count)
            reference_indices = rng.integers(0, 3, size=count)
            sampled_references = references[np.arange(count), reference_indices]
            pred_counts = np.zeros(3, dtype=np.float64)
            gt_counts = np.zeros(3, dtype=np.float64)
            pred_distance = 0.0
            gt_distance = 0.0
            for start in range(0, used, chunk):
                indices = order[start : start + chunk]
                values, distance = r_precision(
                    generated[indices], motion_embeddings[indices], top_k=3
                )
                pred_counts += values
                pred_distance += distance
                values, distance = r_precision(
                    sampled_references[indices], motion_embeddings[indices], top_k=3
                )
                gt_counts += values
                gt_distance += distance
            pred_precision.append(pred_counts / used)
            gt_precision.append(gt_counts / used)
            pred_matching.append(pred_distance / used)
            gt_matching.append(gt_distance / used)

        pred_r = np.stack(pred_precision)
        gt_r = np.stack(gt_precision)
        return {
            "R_precision": pred_r.mean(0).tolist(),
            "R_precision_std": pred_r.std(0).tolist(),
            "Matching_score": float(np.mean(pred_matching)),
            "gt_R_precision": gt_r.mean(0).tolist(),
            "gt_R_precision_std": gt_r.std(0).tolist(),
            "gt_Matching_score": float(np.mean(gt_matching)),
            "semantic_n_samples": int(used),
            "semantic_chunk_size": int(chunk),
            "semantic_n_repeats": int(self.n_repeats),
        }

    def evaluate_records(self, records: Sequence[Dict]) -> dict:
        if not records:
            raise ValueError("No valid M2T prediction records were found.")
        raw_predictions = [str(record["prediction"]).strip() for record in records]
        references = [self._language_references(record) for record in records]
        metrics = compute_coco_caption_metrics(raw_predictions, references)
        if self.compute_bertscore:
            bert_scores = compute_bert_scores(
                raw_predictions,
                references,
                device=self.bert_device,
                model_type=self.bert_model_type,
                batch_size=self.bert_batch_size,
            )
            metrics["Bert_F1"] = bert_scores["rescaled"]
            metrics["Bert_F1_rescaled"] = bert_scores["rescaled"]
            metrics["Bert_F1_raw"] = bert_scores["raw"]
            metrics["Bert_F1_baseline"] = bert_scores["baseline"]
            metrics["Bert_model_type"] = bert_scores["model_type"]
            metrics["Bert_layer"] = bert_scores["layer"]
        else:
            metrics["Bert_F1"] = None
            metrics["Bert_F1_rescaled"] = None
            metrics["Bert_F1_raw"] = None
        metrics.update(
            {
                "n_samples": len(records),
                "reference_policy": "tm2t_first3_repeat_to3",
                "language_reference_mode": self.language_reference_mode,
                "language_prediction_normalization": "none",
                "semantic_prediction_normalization": (
                    "tm2t_spacy_lemma" if self.normalize_predictions else "none"
                ),
            }
        )
        if self.semantic_evaluator is not None:
            semantic_predictions = self._prediction_texts(records)
            metrics.update(self._semantic_metrics(records, semantic_predictions))
        return metrics

    def evaluate(
        self,
        prediction_dir: Union[str, Path],
        *,
        max_samples: Optional[int] = None,
        protocol_manifest: Optional[Union[str, Path]] = None,
        data_root: Optional[Union[str, Path]] = None,
    ) -> dict:
        return self.evaluate_records(
            self.load_prediction_records(
                prediction_dir,
                max_samples=max_samples,
                protocol_manifest=protocol_manifest,
                data_root=data_root,
                io_workers=self.io_workers,
            )
        )


__all__ = ["HumanMLM2TEvaluator"]
