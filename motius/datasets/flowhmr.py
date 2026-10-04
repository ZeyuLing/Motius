"""Official FlowHMR raw-data preprocessing, exposed through Motius DATASETS."""

from torch.utils.data import Dataset, IterableDataset

from motius.registry import DATASETS


@DATASETS.register_module()
class FlowHMRDataset(Dataset):
    def __init__(self, roots, body_model_path, **kwargs):
        from motius.models.flowhmr.vendor.flowhmr.datasets.v2m_generation.train_dataset_raw import V2MTrainDatasetRaw
        kwargs.setdefault("load_hand", True)
        self.dataset = V2MTrainDatasetRaw(roots=roots, smpl_model_path=body_model_path, **kwargs)
        if not len(self.dataset):
            raise ValueError("FlowHMR dataset contains no sample directories")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


@DATASETS.register_module()
class FlowHMRWebDataset(IterableDataset):
    """Resampled official shards; use an iteration-based runner and shuffle=False.

    Accelerate owns rank sharding. WebDataset owns worker sharding; node splitting
    is disabled to avoid sharding the same stream twice.
    """
    def __init__(self, tar_urls, body_model_path, **kwargs):
        super().__init__()
        from motius.models.flowhmr.vendor.flowhmr.datasets.v2m_generation.train_dataset_raw import WDSV2MTrainDatasetRaw
        kwargs.setdefault("load_hand", True)
        kwargs.setdefault("resampled", True)
        kwargs["split_by_node"] = False
        self.source = WDSV2MTrainDatasetRaw(tar_urls=tar_urls,
                     smpl_model_path=body_model_path, **kwargs)

    def __iter__(self):
        return iter(self.source.dataset)
