"""Process-local safety settings for the isolated PromptHMR runtime."""

try:
    import torch
    import torch.utils.data
except ModuleNotFoundError:
    torch = None

if torch is not None:
    # Detectron2's cuDNN convolution path raises SIGFPE with the released
    # PyTorch 2.4 environment on H20. Native CUDA convolution is stable.
    torch.backends.cudnn.enabled = False
    _OriginalDataLoader = torch.utils.data.DataLoader

    class _SafeDataLoader(_OriginalDataLoader):
        def __init__(self, *args, **kwargs):
            kwargs["num_workers"] = 0
            if kwargs.get("batch_size") is not None:
                kwargs["batch_size"] = min(int(kwargs["batch_size"]), 1)
            super().__init__(*args, **kwargs)

    torch.utils.data.DataLoader = _SafeDataLoader
