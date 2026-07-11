def test_core_import_and_registration():
    import motius

    motius.register_all_modules()

    from motius.models import ModelBundle
    from motius.pipelines import BasePipeline
    from motius.trainers import BaseTrainer

    assert ModelBundle is not None
    assert BasePipeline is not None
    assert BaseTrainer is not None
