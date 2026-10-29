import torch
from torch import nn
from motius.trainers.tm2t import TM2TTrainer


def test_motion_to_text_training_uses_summed_loss_and_freezes_codebook():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 8)
            self.head = nn.Linear(8, 16)
        def forward(self, motion, words):
            return self.head(self.embedding(words))
    class Bundle(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = Transformer()
            self.vq_encoder = nn.Linear(2, 2)
            self.quantizer = nn.Linear(2, 2)
            self.text_pad_index = 0
        @property
        def device(self):
            return self.transformer.head.weight.device
    bundle = Bundle()
    trainer = TM2TTrainer(bundle)
    result = trainer.train_step({'motion_tokens': torch.tensor([[1, 2, 3], [3, 2, 1]]),
                                 'text_tokens': torch.tensor([[1, 3, 4, 2], [1, 5, 2, 0]])})
    result['loss'].backward()
    assert torch.isfinite(result['loss'])
    assert bundle.transformer.head.weight.grad is not None
    assert not bundle.vq_encoder.weight.requires_grad
    assert not bundle.quantizer.weight.requires_grad
