"""Evaluation interfaces and released evaluators."""

from motius.evaluation.artifacts import EvaluationArtifactLayout
from motius.evaluation.base_evaluator import BaseEvaluator
from motius.evaluation.dance_to_music import (
    D2MGANBeatScore,
    aggregate_d2mgan_beat_scores,
    d2mgan_beat_bins,
    d2mgan_beat_score,
)
from motius.evaluation.evaluators import (
    HumanMLM2TEvaluator,
    MonocularMetricResult,
    TMRG1Evaluator,
    TMRTextMotionEvaluator,
    evaluate_camera_coordinates,
    evaluate_common_joint_coordinates,
    evaluate_global_coordinates,
)
from motius.evaluation.monocular_capture import (
    MonocularCaptureSample,
    build_3dpw_test_samples,
    build_emdb_samples,
    load_monocular_capture_manifest,
    write_monocular_capture_manifest,
)
from motius.evaluation.monocular_ground_truth import (
    GROUND_TRUTH_REVISION,
    GroundTruthAnnotationError,
    SMPLGeometry,
    materialize_3dpw_ground_truth,
    materialize_emdb_ground_truth,
    materialize_monocular_ground_truth,
)
from motius.evaluation.m2t import (
    HumanML3DM2TSample,
    load_humanml3d_m2t_manifest,
    load_humanml3d_m2t_samples,
    write_humanml3d_m2t_manifest,
)
from motius.evaluation.music_to_dance import (
    AISTPPMusicDanceEvaluator,
    BailandoEvaluator,
    MusicDanceSample,
)
from motius.evaluation.sequential import (
    SequentialCase,
    SequentialSegment,
    evaluate_sequential_cases,
)

__all__ = [
    "AISTPPMusicDanceEvaluator",
    "BaseEvaluator",
    "BailandoEvaluator",
    "D2MGANBeatScore",
    "EvaluationArtifactLayout",
    "GROUND_TRUTH_REVISION",
    "GroundTruthAnnotationError",
    "HumanML3DM2TSample",
    "HumanMLM2TEvaluator",
    "MonocularCaptureSample",
    "MonocularMetricResult",
    "MusicDanceSample",
    "SMPLGeometry",
    "SequentialCase",
    "SequentialSegment",
    "TMRG1Evaluator",
    "TMRTextMotionEvaluator",
    "aggregate_d2mgan_beat_scores",
    "d2mgan_beat_bins",
    "d2mgan_beat_score",
    "build_3dpw_test_samples",
    "build_emdb_samples",
    "evaluate_camera_coordinates",
    "evaluate_common_joint_coordinates",
    "evaluate_global_coordinates",
    "evaluate_sequential_cases",
    "load_monocular_capture_manifest",
    "load_humanml3d_m2t_manifest",
    "load_humanml3d_m2t_samples",
    "materialize_3dpw_ground_truth",
    "materialize_emdb_ground_truth",
    "materialize_monocular_ground_truth",
    "write_humanml3d_m2t_manifest",
    "write_monocular_capture_manifest",
]
