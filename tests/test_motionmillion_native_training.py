from types import SimpleNamespace
import torch
from torch import nn
from motius.trainers.motionmillion import MotionMillionTrainer
from motius.pipelines.motionmillion.sampling import sample_tokens


def test_autoregressive_trainer_backpropagates_and_freezes_tokenizer():
    class AR(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10, 6)
            self.output = nn.Linear(6, 10)
        def forward(self, tokens, features, mask):
            return self.output(self.embedding(tokens))
    class VQVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.ones(()))
            self.vqvae = SimpleNamespace(quantizer=SimpleNamespace(codebook_size=8))
    class Bundle(nn.Module):
        def __init__(self):
            super().__init__()
            self.ar = AR()
            self.vqvae = VQVAE()
            self.text_model = None
        @property
        def device(self):
            return self.ar.output.weight.device
    b = Bundle()
    trainer = MotionMillionTrainer(b, if_maxtest=True)
    result = trainer.train_step({'tokens': torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]]),
                                 'text_features': torch.zeros(2, 3, 6),
                                 'text_mask': torch.ones(2, 3)})
    result['loss'].backward()
    assert torch.isfinite(result['loss'])
    assert b.ar.output.weight.grad is not None
    assert not b.vqvae.p.requires_grad


def test_sampler_stops_when_eos_is_predicted():
    class AR:
        config = SimpleNamespace(vocab_size=10)
        def forward_sample(self, tokens, features, mask):
            logits = torch.zeros(1, 1, 10)
            logits[..., 8] = 1
            return logits
    result = sample_tokens(AR(), torch.zeros(1, 1, 4), torch.ones(1, 1))
    assert result.shape == (1, 1)
