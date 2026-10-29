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


def test_native_mdm_trainer_uses_diffusion_and_backpropagates():
    from motius.trainers.mdm import MDMTrainer

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
        def forward(self, motion, timesteps, y):
            return motion * self.weight

    class Bundle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = Network()
            self._args = SimpleNamespace(diffusion_steps=1000, noise_schedule='linear',
                sigma_small=True, lambda_vel=0., lambda_rcxyz=0., lambda_fc=0.,
                lambda_target_loc=0.)
        @property
        def device(self):
            return self.net.weight.device

    bundle = Bundle()
    trainer = MDMTrainer(bundle)
    result = trainer.train_step(dict(motion=torch.randn(2, 8, 263), motion_lens=[8, 6],
        conditions={'text_tokens': torch.zeros(2, 77, dtype=torch.long)},
        timesteps=[20, 40]))
    result['loss'].backward()
    assert torch.isfinite(result['loss'])
    assert bundle.net.weight.grad is not None
