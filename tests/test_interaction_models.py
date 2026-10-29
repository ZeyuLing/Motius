import numpy as np

from motius.evaluation.evaluators.interhuman_262 import InterHuman262Evaluator
from motius.pipelines.intergen import InterGenPipeline
from motius.pipelines.intermask import InterMaskPipeline


def test_interclip_retrieval_uses_official_fixed_batch_protocol():
    evaluator = InterHuman262Evaluator(
        "unused.safetensors",
        device="cpu",
        retrieval_batch_size=3,
        retrieval_repeats=2,
    )
    embeddings = np.eye(3, dtype=np.float32)
    r_precision, mm_dist = evaluator._retrieval(embeddings, embeddings, seed=11)
    np.testing.assert_allclose(r_precision, [1.0, 1.0, 1.0])
    assert mm_dist == 0.0


def test_interclip_fid_uses_l2_normalized_embeddings():
    evaluator = InterHuman262Evaluator(
        "unused.safetensors",
        device="cpu",
        retrieval_batch_size=3,
        retrieval_repeats=1,
    )
    embeddings = np.eye(3, dtype=np.float32)
    scaled = embeddings * np.asarray([[2.0], [4.0], [8.0]], dtype=np.float32)
    evaluator.embed_pack = lambda pack: (embeddings, pack)
    results = evaluator.evaluate_packs(
        scaled,
        {"Scaled": scaled},
    )

    assert results["Scaled"]["fid_embedding_space"] == "l2_normalized"
    assert abs(results["Scaled"]["fid"]) < 1e-8


def test_interaction_components_are_registered():
    from motius.models.intergen import InterGenBundle
    from motius.models.intermask import InterMaskBundle
    from motius.registry import EVALUATORS, MODEL_BUNDLES, PIPELINES

    assert PIPELINES.get("InterGenPipeline") is InterGenPipeline
    assert PIPELINES.get("InterMaskPipeline") is InterMaskPipeline
    assert MODEL_BUNDLES.get("InterGenBundle") is InterGenBundle
    assert MODEL_BUNDLES.get("InterMaskBundle") is InterMaskBundle
    assert EVALUATORS.get("InterHuman262Evaluator") is InterHuman262Evaluator
