def test_core_import_and_registration():
    import motius

    motius.register_all_modules()

    from motius.models import ModelBundle
    from motius.pipelines import BasePipeline
    from motius.trainers import BaseTrainer

    assert ModelBundle is not None
    assert BasePipeline is not None
    assert BaseTrainer is not None


def test_mdm_pipeline_import_and_registration():
    from motius.models.mdm import MDMBundle
    from motius.pipelines.mdm import MDMPipeline
    from motius.registry import MODEL_BUNDLES, PIPELINES

    assert MDMPipeline.BUNDLE_CLS == "motius.models.mdm.MDMBundle"
    assert PIPELINES.get("MDMPipeline") is MDMPipeline
    assert MODEL_BUNDLES.get("MDMBundle") is MDMBundle


def test_condmdi_pipeline_import_and_registration():
    from motius.models.condmdi import CondMDIBundle
    from motius.pipelines.condmdi import CondMDIPipeline
    from motius.registry import MODEL_BUNDLES, PIPELINES

    assert CondMDIPipeline.BUNDLE_CLS == "motius.models.condmdi.CondMDIBundle"
    assert PIPELINES.get("CondMDIPipeline") is CondMDIPipeline
    assert MODEL_BUNDLES.get("CondMDIBundle") is CondMDIBundle


def test_omnicontrol_pipeline_import_and_registration():
    import pytest

    pytest.importorskip("clip")
    from motius.models.omnicontrol import OmniControlBundle
    from motius.pipelines.omnicontrol import OmniControlPipeline
    from motius.registry import MODEL_BUNDLES, PIPELINES

    assert OmniControlPipeline.BUNDLE_CLS == "motius.models.omnicontrol.OmniControlBundle"
    assert PIPELINES.get("OmniControlPipeline") is OmniControlPipeline
    assert MODEL_BUNDLES.get("OmniControlBundle") is OmniControlBundle
