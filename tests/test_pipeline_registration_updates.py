from motius.pipelines.auto import _trusted_pipeline_class_path, _pipeline_path_from_manifest


def test_native_pipelines_resolve_through_allowlist():
    expected = {
        'MoGenDiTPipeline': 'motius.pipelines.mogendit.MoGenDiTPipeline',
        'PromptHMRPipeline': 'motius.pipelines.prompthmr.PromptHMRPipeline',
        'MotionStreamerMatchingPipeline': 'motius.pipelines.motionstreamer_matching.MotionStreamerMatchingPipeline',
    }
    for name, path in expected.items():
        assert _trusted_pipeline_class_path(path) == path
        assert _pipeline_path_from_manifest({"_class_name": name}) == path
