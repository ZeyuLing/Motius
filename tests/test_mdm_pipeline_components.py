"""Exercise the MDM pipeline's own scheduler and collation, without CLIP weights."""
from types import SimpleNamespace

import torch

from motius.pipelines.mdm import MDMPipeline


class TextNetwork(torch.nn.Module):
    dataset = "humanml"
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.data_rep = "hml_vec"

    def encode_text(self, captions):
        return torch.zeros(len(captions), 1)


class RecordingDiffusion:
    def p_sample_loop(self, model, shape, model_kwargs, **kwargs):
        self.shape = shape
        self.lengths = model_kwargs["y"]["lengths"].tolist()
        return torch.zeros(shape)


def test_scheduler_is_injected_into_pipeline_and_lengths_are_collated():
    diffusion = RecordingDiffusion()
    bundle = SimpleNamespace(net=TextNetwork(), guidance_param=1.0,
                             eval=lambda: None, mean=torch.zeros(263), std=torch.ones(263))
    pipeline = MDMPipeline(bundle, diffusion=diffusion)
    outputs = pipeline.infer_t2m(["walk", "jump"], [40, 52])
    assert [value.shape for value in outputs] == [(40, 263), (52, 263)]
    assert diffusion.shape == (2, 263, 1, 52)
    assert diffusion.lengths == [40, 52]
