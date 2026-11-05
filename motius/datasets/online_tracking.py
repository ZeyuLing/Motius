"""Process-local streams of fresh vector-environment rollouts for the runner."""
from torch.utils.data import IterableDataset,get_worker_info


class OnlineTrackingRollouts(IterableDataset):
    def __init__(self,collector,steps=24):
        super().__init__();self.collector=collector;self.steps=int(steps)

    def __iter__(self):
        if get_worker_info() is not None:raise RuntimeError('Online simulation requires num_workers=0')
        while True:yield self.collector.collect(self.steps)
