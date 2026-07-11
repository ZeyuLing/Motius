def test_core_import_and_registration():
    import motius

    motius.register_all_modules()

    from hftrainer.models import ModelBundle
    from hftrainer.pipelines import BasePipeline
    from hftrainer.trainers import BaseTrainer

    assert ModelBundle is not None
    assert BasePipeline is not None
    assert BaseTrainer is not None
