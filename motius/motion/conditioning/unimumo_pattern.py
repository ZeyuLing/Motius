from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class DelayedPattern:
    timesteps: int
    delays: tuple[int, ...]

    @property
    def num_codebooks(self) -> int:
        return len(self.delays)

    @property
    def sequence_length(self) -> int:
        return self.timesteps + max(self.delays) + 1

    def build(self, codes: torch.Tensor, *, special_token: int, valid_only: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
        batch, codebooks, timesteps = codes.shape
        if codebooks != self.num_codebooks or timesteps != self.timesteps:
            raise ValueError('codes do not match delayed pattern')
        sequence_length = self.timesteps + 1 if valid_only else self.sequence_length
        sequence = torch.full((batch, codebooks, sequence_length), special_token, dtype=codes.dtype, device=codes.device)
        valid = torch.zeros((codebooks, sequence_length), dtype=torch.bool, device=codes.device)
        for codebook, delay in enumerate(self.delays):
            start = 1 + delay
            end = min(start + timesteps, sequence_length)
            count = max(0, end - start)
            if count:
                sequence[:, codebook, start:end] = codes[:, codebook, :count]
                valid[codebook, start:end] = True
        return (sequence, valid)

    def revert(self, sequence: torch.Tensor) -> torch.Tensor:
        values = []
        for codebook, delay in enumerate(self.delays):
            start = 1 + delay
            values.append(sequence[:, codebook, start:start + self.timesteps])
        return torch.stack(values, dim=1)


def restore_prediction_logits(logits,timesteps,delays):
    """Restore next-token logits and their valid mask from a delayed sequence."""
    batch,books,steps,classes=logits.shape
    restored=logits.new_full((batch,books,timesteps,classes),float('nan'))
    mask=torch.zeros((batch,books,timesteps),dtype=torch.bool,device=logits.device)
    for book,delay in enumerate(delays):
        count=max(0,min(timesteps,steps-1-delay))
        restored[:,book,:count]=logits[:,book,delay:delay+count]
        mask[:,book,:count]=True
    return restored,mask
