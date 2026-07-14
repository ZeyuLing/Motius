import json

import numpy as np
import pytest

from tools.eval_t2m_joint_evaluator import load_protocol


def test_load_protocol_uses_selected_caption_and_short_prediction_name(tmp_path):
    dataset = tmp_path / "dataset"
    predictions = tmp_path / "predictions"
    (dataset / "motions").mkdir(parents=True)
    (dataset / "splits").mkdir()
    predictions.mkdir()
    annotation = {
        "h3dtest_001840": {
            "path": "h3dtest_001840",
            "annotations": [{"text": "a selected caption"}],
        }
    }
    (dataset / "annotations.json").write_text(json.dumps(annotation))
    (dataset / "splits" / "humanml3d_test.txt").write_text("h3dtest_001840\n")
    reference = np.zeros((8, 66), dtype=np.float32)
    prediction = np.ones((7, 22, 3), dtype=np.float32)
    np.save(dataset / "motions" / "h3dtest_001840.npy", reference)
    np.save(predictions / "001840.npy", prediction)

    captions, predicted, references, keyids = load_protocol(
        dataset, "humanml3d_test", predictions
    )

    assert captions == ["a selected caption"]
    assert keyids == ["h3dtest_001840"]
    assert predicted[0].shape == (7, 66)
    np.testing.assert_array_equal(references[0], reference)


def test_load_protocol_rejects_ambiguous_caption_selection(tmp_path):
    dataset = tmp_path / "dataset"
    predictions = tmp_path / "predictions"
    (dataset / "motions").mkdir(parents=True)
    (dataset / "splits").mkdir()
    predictions.mkdir()
    annotation = {
        "h3dtest_000001": {
            "path": "h3dtest_000001",
            "annotations": [{"text": "first"}, {"text": "second"}],
        }
    }
    (dataset / "annotations.json").write_text(json.dumps(annotation))
    (dataset / "splits" / "humanml3d_test.txt").write_text("h3dtest_000001\n")

    with pytest.raises(ValueError, match="exactly one selected caption"):
        load_protocol(dataset, "humanml3d_test", predictions)
