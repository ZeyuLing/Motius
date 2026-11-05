"""Prepared DART normalization and licensed body geometry owned by training."""

from pathlib import Path
import pickle
from motius.motion.representation.dart import PrimitiveUtility


class DARTTrainingProcessor:
    def __init__(self, bundle):
        args = bundle.vae_args.data_args
        self.history_length = int(args.history_length)
        self.future_length = int(args.future_length)
        self.num_primitive = 1
        path = (
            Path(bundle.artifact_dir)
            / args.data_dir
            / f"mean_std_h{self.history_length}_f{self.future_length}.pkl"
        )
        with path.open("rb") as stream:
            self.mean, self.std = pickle.load(stream)
        self.primitive_utility = PrimitiveUtility(
            device="cpu", body_type=args.body_type
        )

    def to(self, device):
        self.mean, self.std = self.mean.to(device), self.std.to(device)
        self.primitive_utility.device = device
        self.primitive_utility.bm_male.to(device)
        self.primitive_utility.bm_female.to(device)
        return self

    def denormalize(self, motion):
        return motion * self.std.to(device=motion.device) + self.mean.to(
            device=motion.device
        )

    def tensor_to_dict(self, motion):
        return self.primitive_utility.tensor_to_dict(motion)
