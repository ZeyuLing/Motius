from torch import Tensor, nn
from os.path import join as pjoin
from .mr import MRMetrics
from .t2m import TM2TMetrics
from .mm import MMMetrics
from .m2m import PredMetrics

try:
    from .m2t import M2TMetrics
except Exception:
    M2TMetrics = None

try:
    from .tmr import TMRMetrics
except Exception:
    TMRMetrics = None


class BaseMetrics(nn.Module):
    def __init__(self, cfg, datamodule, debug, **kwargs) -> None:
        super().__init__()

        njoints = datamodule.njoints
        # cfg.METRIC.TM2T.t2m_textencoder.params['dataset'] = datamodule.name

        data_name = datamodule.name
        print('in BaseMetrics, data_name:', data_name)
        if data_name in ["humanml3d", "kit", 'motionx', 'tomato']:
            if 'TM2TMetrics' in cfg.METRIC.TYPE:
                self.TM2TMetrics = TM2TMetrics(
                    cfg=cfg,
                    dataname=data_name,
                    diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                    njoints=njoints,
                )
            if 'M2TMetrics' in cfg.METRIC.TYPE or cfg.model.params.task =='m2t':
                if M2TMetrics is None:
                    raise ImportError("M2TMetrics requested, but its optional dependencies are unavailable")
                self.M2TMetrics = M2TMetrics(
                    cfg=cfg,
                    dataname=data_name,
                    w_vectorizer=datamodule.hparams.w_vectorizer,
                    diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP)
            self.MMMetrics = MMMetrics(
                cfg=cfg,
                dataname=data_name,
                mm_num_times=cfg.METRIC.MM_NUM_TIMES,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                njoints=njoints,
            )
            if 'TemosMetric' in cfg.METRIC.TYPE:
                from .compute import ComputeMetrics
                self.TemosMetric = ComputeMetrics(
                    njoints=njoints,
                    jointstype=cfg.DATASET.JOINT_TYPE,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            if 'TMRMetrics' in cfg.METRIC.TYPE:
                if TMRMetrics is None:
                    raise ImportError("TMRMetrics requested, but its optional dependencies are unavailable")
                self.TMRMetrics = TMRMetrics(
                    cfg=cfg,
                    dataname=data_name,
                    diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                    threshold_selfsim_metrics=0.95
                )

        if 'MRMetrics' in cfg.METRIC.TYPE:
            self.MRMetrics = MRMetrics(
                njoints=njoints,
                jointstype=cfg.DATASET.JOINT_TYPE,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            )
        if 'PredMetrics' in cfg.METRIC.TYPE:
            self.PredMetrics = PredMetrics(
                cfg=cfg,
                njoints=njoints,
                jointstype=cfg.DATASET.JOINT_TYPE,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                task=cfg.model.params.task,
            )
