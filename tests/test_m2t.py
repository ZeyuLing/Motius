from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from motius.evaluation.m2t import (
    HUMANML3D_MAX_FRAMES,
    compute_bert_scores,
    load_humanml3d_m2t_manifest,
    load_humanml3d_m2t_samples,
    official_reference_count,
    write_prediction_records,
    write_humanml3d_m2t_manifest,
)
from motius.evaluation.evaluators.humanml3d_m2t import HumanMLM2TEvaluator
from motius.pipelines.motiongpt import MotionGPTPipeline
from motius.pipelines.motiongpt3 import MotionGPT3Pipeline
from motius.pipelines.tm2t import TM2TPipeline
from motius.pipelines.vermo import motion135_to_vermo138


def test_official_reference_count_matches_tm2t_policy():
    assert official_reference_count(["a"]) == ("a", "a", "a")
    assert official_reference_count(["a", "b"]) == ("a", "b", "a")
    assert official_reference_count(["a", "b", "c", "d"]) == ("a", "b", "c")
    with pytest.raises(ValueError):
        official_reference_count([])


def test_humanml3d_m2t_loader_includes_temporal_subclips(tmp_path):
    root = tmp_path / "HumanML3D"
    (root / "new_joint_vecs").mkdir(parents=True)
    (root / "texts").mkdir()
    np.save(root / "new_joint_vecs" / "000001.npy", np.zeros((160, 263), np.float32))
    (root / "test.txt").write_text("000001\n", encoding="utf-8")
    (root / "texts" / "000001.txt").write_text(
        "a person walks#a/DET person/NOUN walk/VERB#0#0\n"
        "someone is walking#someone/NOUN be/VERB walk/VERB#0#0\n"
        "a person turns#person/NOUN turn/VERB#2#5\n",
        encoding="utf-8",
    )

    samples = load_humanml3d_m2t_samples(root)

    assert [sample.sample_id for sample in samples] == [
        "000001_2.000000_5.000000",
        "000001",
    ]
    assert samples[1].length == 160
    assert samples[1].token_references == ("a person walk", "someone be walk")
    assert samples[0].start_frame == 40
    assert samples[0].end_frame == 100
    assert samples[0].load_motion().shape == (60, 263)

    assert write_prediction_records(tmp_path / "pred", [(samples[1], "walking")]) == 1
    record = json.loads(
        (tmp_path / "pred" / "predictions" / "000001.json").read_text(encoding="utf-8")
    )
    assert record["references"] == ["a person walks", "someone is walking", "a person walks"]
    assert record["start_frame"] == 0
    assert record["end_frame"] == 160


def test_humanml3d_m2t_loader_matches_official_duplicate_overwrite(tmp_path):
    root = tmp_path / "HumanML3D"
    (root / "new_joint_vecs").mkdir(parents=True)
    (root / "texts").mkdir()
    for source_id in ("000001", "000002"):
        np.save(
            root / "new_joint_vecs" / f"{source_id}.npy",
            np.zeros((160, 263), np.float32),
        )
    (root / "test.txt").write_text("000001\n000002\n", encoding="utf-8")
    (root / "texts" / "000001.txt").write_text(
        "first turn#first/ADJ turn/VERB#1#4\n"
        "final turn#final/ADJ turn/VERB#1#4\n"
        "full motion#full/ADJ motion/NOUN#0#0\n",
        encoding="utf-8",
    )
    (root / "texts" / "000002.txt").write_text(
        "second full#second/ADJ full/NOUN#0#0\n", encoding="utf-8"
    )

    samples = load_humanml3d_m2t_samples(root)

    # Official TM2T reports len(data_dict), indexes name_list, and therefore
    # repeats the overwritten temporal entry while dropping the final name.
    assert [sample.sample_id for sample in samples] == [
        "000001_1.000000_4.000000",
        "000001_1.000000_4.000000",
        "000001",
    ]
    assert samples[0].references == ("final turn",)
    assert samples[1].references == ("final turn",)


def test_m2t_evaluator_manifest_keeps_empty_and_duplicate_predictions(tmp_path):
    root = tmp_path / "HumanML3D"
    (root / "new_joint_vecs").mkdir(parents=True)
    (root / "texts").mkdir()
    np.save(root / "new_joint_vecs" / "000001.npy", np.zeros((80, 263), np.float32))
    (root / "test.txt").write_text("000001\n", encoding="utf-8")
    (root / "texts" / "000001.txt").write_text(
        "first#first/NOUN#1#3\nsecond#second/NOUN#1#3\nfull#full/NOUN#0#0\n",
        encoding="utf-8",
    )
    samples = load_humanml3d_m2t_samples(root)
    manifest = write_humanml3d_m2t_manifest(
        samples, tmp_path / "protocol.json", data_root=root
    )
    write_prediction_records(tmp_path / "pred", [(samples[0], "")])

    records = HumanMLM2TEvaluator.load_prediction_records(
        tmp_path / "pred", protocol_manifest=manifest
    )

    assert len(records) == 2
    assert records[0]["prediction"] == ""
    assert records[0]["metric_references"] == ["second", "second", "second"]


def test_m2t_language_reference_mode_is_explicit(monkeypatch):
    captured = []

    def fake_coco(predictions, references):
        captured.append((predictions, references))
        return {"Bleu_1": 0.0}

    monkeypatch.setattr(
        "motius.evaluation.evaluators.humanml3d_m2t.compute_coco_caption_metrics",
        fake_coco,
    )
    record = {
        "prediction": "A person is walking.",
        "references": ["Someone walks forward."],
        "token_references": ["someone walk forward"],
        "metric_references": ["someone walk forward"] * 3,
    }

    token_evaluator = HumanMLM2TEvaluator(
        compute_bertscore=False, language_reference_mode="token"
    )
    token_metrics = token_evaluator.evaluate_records([record])
    raw_evaluator = HumanMLM2TEvaluator(
        compute_bertscore=False, language_reference_mode="raw"
    )
    raw_metrics = raw_evaluator.evaluate_records([record])

    assert captured[0][1] == [["someone walk forward"] * 3]
    assert captured[1][1] == [["Someone walks forward."] * 3]
    assert token_metrics["language_reference_mode"] == "token"
    assert raw_metrics["language_reference_mode"] == "raw"


def test_m2t_rejects_unknown_language_reference_mode():
    with pytest.raises(ValueError, match="language_reference_mode"):
        HumanMLM2TEvaluator(language_reference_mode="mixed")


def test_bert_score_exposes_raw_and_rescaled_scales(monkeypatch):
    def fake_score(*args, **kwargs):
        assert kwargs["rescale_with_baseline"] is False
        values = torch.tensor([0.90, 0.88], dtype=torch.float32)
        return values, values, values

    monkeypatch.setattr("bert_score.score", fake_score)
    monkeypatch.setattr(
        "motius.evaluation.m2t._bert_score_f1_baseline",
        lambda **kwargs: (0.83, "roberta-large", 17),
    )

    scores = compute_bert_scores(
        ["first", "second"],
        [["reference"], ["reference"]],
        device="cpu",
    )

    assert scores["raw"] == pytest.approx(0.89)
    assert scores["rescaled"] == pytest.approx((0.89 - 0.83) / 0.17)
    assert scores["baseline"] == 0.83
    assert scores["model_type"] == "roberta-large"
    assert scores["layer"] == 17


def test_m2t_semantic_retrieval_uses_text_queries(monkeypatch, tmp_path):
    class FakeSemanticEvaluator:
        @staticmethod
        def encode_texts(texts):
            value = 10.0 if len(texts) == 4 else 20.0
            return np.full((len(texts), 2), value, dtype=np.float32)

        @staticmethod
        def encode_motions(motions):
            return np.full((len(motions), 2), 30.0, dtype=np.float32)

    motion_path = tmp_path / "motion.npy"
    np.save(motion_path, np.zeros((40, 263), dtype=np.float32))
    records = [
        {
            "motion_path": str(motion_path),
            "start_frame": 0,
            "end_frame": 40,
            "metric_references": ["first", "second", "third"],
        }
        for _ in range(4)
    ]
    calls = []

    def fake_r_precision(text, motion, top_k):
        calls.append((text.copy(), motion.copy(), top_k))
        return np.full(top_k, len(text), dtype=np.float64), 0.0

    monkeypatch.setattr(
        "motius.evaluation.evaluators.humanml3d_m2t.r_precision",
        fake_r_precision,
    )
    evaluator = HumanMLM2TEvaluator(
        semantic_evaluator=FakeSemanticEvaluator(),
        chunk_size=4,
        n_repeats=1,
        compute_bertscore=False,
    )

    evaluator._semantic_metrics(records, ["prediction"] * 4)

    assert len(calls) == 2
    np.testing.assert_allclose(calls[0][0], 10.0)
    np.testing.assert_allclose(calls[0][1], 30.0)
    np.testing.assert_allclose(calls[1][0], 20.0)
    np.testing.assert_allclose(calls[1][1], 30.0)
    assert calls[0][2] == 3


def test_humanml3d_m2t_loader_keeps_199_frames_and_truncates_to_196(tmp_path):
    root = tmp_path / "HumanML3D"
    (root / "new_joint_vecs").mkdir(parents=True)
    (root / "texts").mkdir()
    np.save(root / "new_joint_vecs" / "000002.npy", np.zeros((199, 263), np.float32))
    (root / "test.txt").write_text("000002\n", encoding="utf-8")
    (root / "texts" / "000002.txt").write_text(
        "a person walks#a/DET person/NOUN walk/VERB#0#0\n",
        encoding="utf-8",
    )

    samples = load_humanml3d_m2t_samples(root)

    assert len(samples) == 1
    assert samples[0].length == HUMANML3D_MAX_FRAMES
    assert samples[0].load_motion().shape == (HUMANML3D_MAX_FRAMES, 263)


def test_humanml3d_m2t_manifest_round_trip_supports_relocation(tmp_path):
    root = tmp_path / "source" / "HumanML3D"
    relocated = tmp_path / "relocated" / "HumanML3D"
    for data_root in (root, relocated):
        (data_root / "new_joint_vecs").mkdir(parents=True)
        (data_root / "texts").mkdir()
        np.save(
            data_root / "new_joint_vecs" / "000003.npy",
            np.zeros((80, 263), np.float32),
        )
        (data_root / "test.txt").write_text("000003\n", encoding="utf-8")
        (data_root / "texts" / "000003.txt").write_text(
            "a person walks#a/DET person/NOUN walk/VERB#0#0\n",
            encoding="utf-8",
        )

    samples = load_humanml3d_m2t_samples(root)
    manifest = write_humanml3d_m2t_manifest(
        samples, tmp_path / "protocol.json", data_root=root
    )
    restored = load_humanml3d_m2t_manifest(manifest, data_root=relocated)

    assert len(restored) == 1
    assert restored[0].motion_path == relocated.resolve() / "new_joint_vecs/000003.npy"
    assert restored[0].references == ("a person walks",)
    assert restored[0].load_motion().flags.writeable


class _FakeVAE:
    def __init__(self):
        self.inputs = []

    def encode(self, value):
        self.inputs.append(value.detach().cpu())
        return torch.tensor([[7, 8]], device=value.device).repeat(
            value.shape[0], 1
        ), None


class _FakeLM:
    def __init__(self):
        self.kwargs = None

    def generate_conditional(self, **kwargs):
        self.kwargs = kwargs
        return [" a person walks "] * len(kwargs["motion_tokens"])


class _FakeBundle:
    def __init__(self):
        self.vae = _FakeVAE()
        self.lm = _FakeLM()
        self.mean = torch.tensor([1.0, 2.0])
        self.std = torch.tensor([2.0, 4.0])

    @property
    def device(self):
        return self.mean.device

    def eval(self):
        return self


def test_motiongpt_m2t_normalizes_before_tokenization():
    bundle = _FakeBundle()
    pipeline = MotionGPTPipeline(bundle)
    motion = np.array([[3.0, 6.0], [5.0, 10.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="HML3D-263"):
        pipeline.infer_m2t([motion])

    bundle.mean = torch.zeros(263)
    bundle.std = torch.full((263,), 2.0)
    motion = np.full((3, 263), 4.0, dtype=np.float32)
    result = pipeline.infer_m2t([motion], lengths=[2])

    assert result == ["a person walks"]
    torch.testing.assert_close(bundle.vae.inputs[0], torch.full((1, 2, 263), 2.0))
    assert bundle.lm.kwargs["task"] == "m2t"
    assert bundle.lm.kwargs["lengths"] == [2]


def test_motiongpt_m2t_reproduces_official_batch_padding():
    bundle = _FakeBundle()
    bundle.mean = torch.zeros(263)
    bundle.std = torch.full((263,), 2.0)
    pipeline = MotionGPTPipeline(bundle)
    motions = [
        np.full((3, 263), 4.0, dtype=np.float32),
        np.full((2, 263), 6.0, dtype=np.float32),
    ]

    result = pipeline.infer_m2t(motions, pad_to_batch_max=True)

    assert result == ["a person walks", "a person walks"]
    expected = torch.zeros((2, 3, 263))
    expected[0] = 2.0
    expected[1, :2] = 3.0
    torch.testing.assert_close(bundle.vae.inputs[0], expected)
    assert bundle.lm.kwargs["lengths"] == [2, 2]


def test_motiongpt_m2t_default_is_batch_invariant():
    bundle = _FakeBundle()
    bundle.mean = torch.zeros(263)
    bundle.std = torch.full((263,), 2.0)
    pipeline = MotionGPTPipeline(bundle)
    motions = [
        np.full((3, 263), 4.0, dtype=np.float32),
        np.full((2, 263), 6.0, dtype=np.float32),
    ]

    result = pipeline.infer_m2t(motions)

    assert result == ["a person walks", "a person walks"]
    assert [tuple(value.shape) for value in bundle.vae.inputs] == [
        (1, 3, 263),
        (1, 2, 263),
    ]


class _FakeMotionGPT3LM:
    def __init__(self):
        self.kwargs = None

    def generate_conditional(self, **kwargs):
        self.kwargs = kwargs
        return [" a person turns ", " someone jumps "]


class _FakeMotionGPT3Bundle:
    def __init__(self):
        self.mean = torch.ones(263)
        self.std = torch.full((263,), 2.0)
        self.model = type(
            "Model",
            (),
            {"lm": _FakeMotionGPT3LM(), "vae": object()},
        )()

    @property
    def device(self):
        return self.mean.device

    def eval(self):
        return self


def test_motiongpt3_m2t_builds_padded_normalized_batch():
    bundle = _FakeMotionGPT3Bundle()
    pipeline = MotionGPT3Pipeline(bundle)
    motions = [
        np.full((3, 263), 5.0, dtype=np.float32),
        np.full((2, 263), 3.0, dtype=np.float32),
    ]

    result = pipeline.infer_m2t(motions)

    assert result == ["a person turns", "someone jumps"]
    kwargs = bundle.model.lm.kwargs
    assert kwargs["task"] == "m2t"
    assert kwargs["lengths"] == [3, 2]
    assert kwargs["motion_encode_net"] is bundle.model.vae
    expected = torch.zeros((2, 3, 263))
    expected[0] = 2.0
    expected[1, :2] = 1.0
    torch.testing.assert_close(kwargs["motion_feats"], expected)


class _FakeTM2TVQ:
    def __init__(self):
        self.value = None

    def __call__(self, value):
        self.value = value
        return torch.zeros((1, 2, 4), device=value.device)


class _FakeTM2TQuantizer:
    def map2index(self, value):
        return torch.tensor([11, 12], device=value.device)


class _FakeTM2TTranslator:
    def __init__(self):
        self.tokens = None

    def translate_sentence(self, tokens):
        self.tokens = tokens
        return [100, 21, 22, 101]


class _FakeTM2TVocabulary:
    def token(self, index):
        return {21: "a", 22: "walk"}[index]


class _FakeTM2TBundle:
    def __init__(self):
        self.mean = torch.ones(263)
        self.std = torch.full((263,), 2.0)
        self.vq_encoder = _FakeTM2TVQ()
        self.quantizer = _FakeTM2TQuantizer()
        self.translator = _FakeTM2TTranslator()
        self.vocabulary = _FakeTM2TVocabulary()
        self.motion_start_index = 1024
        self.motion_end_index = 1025
        self.motion_pad_index = 1026

    @property
    def device(self):
        return self.mean.device

    def eval(self):
        return self


def test_tm2t_m2t_tokenizes_hml263_with_official_layout():
    bundle = _FakeTM2TBundle()
    pipeline = TM2TPipeline(bundle)
    motion = np.full((43, 263), 5.0, dtype=np.float32)

    result = pipeline.infer_m2t([motion])

    assert result == ["a walk"]
    assert bundle.vq_encoder.value.shape == (1, 40, 259)
    torch.testing.assert_close(bundle.vq_encoder.value, torch.full((1, 40, 259), 2.0))
    tokens = bundle.translator.tokens[0].tolist()
    assert tokens[:4] == [1024, 11, 12, 1025]
    assert len(tokens) == 55
    assert set(tokens[4:]) == {1026}


def test_vermo_motion135_conversion_uses_abs_rel_and_column_rot6d():
    motion = np.zeros((2, 135), dtype=np.float32)
    motion[:, 3:] = np.tile(
        np.array([1, 0, 0, 1, 0, 0], dtype=np.float32), 22
    )
    motion[0, :3] = [1, 2, 3]
    motion[1, :3] = [4, 6, 8]

    converted = motion135_to_vermo138(motion)

    assert converted.shape == (2, 138)
    np.testing.assert_allclose(converted[:, :3], motion[:, :3])
    np.testing.assert_allclose(converted[:, 3:6], [[0, 0, 0], [3, 4, 5]])
    np.testing.assert_allclose(converted[0, 6:12], [1, 0, 0, 0, 1, 0])
