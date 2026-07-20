import sys
from typing import Union, Any, Tuple, Dict

import os
import yaml
from einops import rearrange
from torch import nn
import torch

from .encodec import EncodecFeatures
from .modules.istft_head import ISTFTHead
from .modules.utils import save_audio
from .modules.vocos import VocosBackbone
from motius.registry import MODELS


def _import_torchaudio():
    try:
        import torchaudio
    except Exception as exc:
        raise RuntimeError(
            "WavTokenizer audio file loading requires torchaudio and matching "
            "torch/torchaudio binaries."
        ) from exc
    return torchaudio


def find_config_and_checkpoint(repo):
    config_path = None
    checkpoint_path = None

    for root, dirs, files in os.walk(repo):
        for file in files:
            if file.endswith(".yaml"):
                config_path = os.path.join(root, file)
            elif file.endswith(".ckpt"):
                checkpoint_path = os.path.join(root, file)

            if config_path and checkpoint_path:
                return config_path, checkpoint_path

    return config_path, checkpoint_path


# Remap legacy mmotion paths to their Motius equivalents.
_CLASS_PATH_REMAPS = {
    "mmotion.models.autoencoders.wavtokenizer": "motius.models.vermo.wavtokenizer",
}


def _remap_class_path(class_path: str) -> str:
    """Remap legacy class paths (for example, ``mmotion.*``) to Motius."""
    for old_prefix, new_prefix in _CLASS_PATH_REMAPS.items():
        if class_path.startswith(old_prefix):
            return new_prefix + class_path[len(old_prefix):]
    return class_path


def instantiate_class(args: Union[Any, Tuple[Any, ...]], init: Dict[str, Any]) -> Any:
    """Instantiates a class with the given args and init.

    Args:
        args: Positional arguments required for instantiation.
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.
    """
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)
    class_path = _remap_class_path(init["class_path"])
    class_module, class_name = class_path.rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(*args, **kwargs)


@MODELS.register_module(force=True)
class WavTokenizer(nn.Module):
    """
    The Vocos class represents a Fourier-based neural vocoder for audio synthesis.
    This class is primarily designed for inference, with support for loading from pretrained
    model checkpoints. It consists of three main components: a feature extractor,
    a backbone, and a head.
    """

    def __init__(
        self,
        pretrained: str = None,
        feature_extractor: EncodecFeatures = None,
        backbone: VocosBackbone = None,
        head: ISTFTHead = None,
    ):
        super().__init__()
        if pretrained is None:
            self.feature_extractor = feature_extractor
            self.backbone = backbone
            self.head = head
        else:
            config_path, checkpoint_path = find_config_and_checkpoint(pretrained)
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            self.feature_extractor: EncodecFeatures = instantiate_class(
                args=(), init=config["model"]["init_args"]["feature_extractor"]
            )
            self.backbone: VocosBackbone = instantiate_class(
                args=(), init=config["model"]["init_args"]["backbone"]
            )
            self.head: ISTFTHead = instantiate_class(
                args=(), init=config["model"]["init_args"]["head"]
            )

            state_dict_raw = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )["state_dict"]
            state_dict = dict()
            for k, v in state_dict_raw.items():
                if (
                    k.startswith("backbone.")
                    or k.startswith("head.")
                    or k.startswith("feature_extractor.")
                ):
                    state_dict[k] = v

            self.load_state_dict(state_dict)
        self._disable_rnn_flatten_parameters()

    def _disable_rnn_flatten_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
                module.flatten_parameters = lambda: None

    @classmethod
    def from_hparams(cls, config_path: str) -> "Vocos":
        """
        Class method to create a new Vocos model instance from hyperparameters stored in a yaml configuration file.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        feature_extractor = instantiate_class(
            args=(), init=config["model"]["init_args"]["feature_extractor"]
        )
        backbone = instantiate_class(
            args=(), init=config["model"]["init_args"]["backbone"]
        )
        head = instantiate_class(args=(), init=config["model"]["init_args"]["head"])
        model = cls(feature_extractor=feature_extractor, backbone=backbone, head=head)
        return model

    @classmethod
    def from_pretrained(self, config_path, model_path):
        """
        Class method to create a new Vocos model instance from a pre-trained model stored in the Hugging Face model hub.
        """
        model = self.from_hparams(config_path)
        state_dict_raw = torch.load(model_path, map_location="cpu", weights_only=True)[
            "state_dict"
        ]
        state_dict = dict()
        for k, v in state_dict_raw.items():
            if (
                k.startswith("backbone.")
                or k.startswith("head.")
                or k.startswith("feature_extractor.")
            ):
                state_dict[k] = v

        model.load_state_dict(state_dict)
        model.eval()
        return model

    def forward(self, audio_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Method to run a copy-synthesis from audio waveform. The feature extractor first processes the audio input,
        which is then passed through the backbone and the head to reconstruct the audio output.

        Args:
            audio_input (Tensor): The input tensor representing the audio waveform of shape (B, T),
                                        where B is the batch size and L is the waveform length.


        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        features, _, _ = self.feature_extractor(audio_input, **kwargs)  # 0818
        audio_output = self.decode(features, **kwargs)
        return audio_output

    # 0818
    def encode_train(
        self, audio_input: torch.Tensor, **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features, discrete_codes, _ = self.feature_extractor(audio_input, **kwargs)
        return features, discrete_codes

    # 0818
    def encode(
        self, audio_input: torch.Tensor, **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param audio_input: b t
        :param kwargs:
        :return: b d n, b n
        """
        bandwidth_id = torch.tensor([0]).to(audio_input.device)
        features, discrete_codes, _ = self.feature_extractor.infer(
            audio_input, bandwidth_id=bandwidth_id, **kwargs
        )
        # remove the num_quantizers dim
        discrete_codes = discrete_codes.squeeze(0)
        return features, discrete_codes

    def decode(
        self, features_input: torch.Tensor, is_idx: bool = False, **kwargs: Any
    ) -> torch.Tensor:
        """
        Method to decode audio waveform from already calculated features. The features input is passed through
        the backbone and the head to reconstruct the audio output.

        Args:
            features_input (Tensor): The input tensor of features of shape (B, C, L), where B is the batch size,
                                     C denotes the feature dimension, and L is the sequence length.

        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        if is_idx:
            features_input = self.codes_to_features(features_input)
        bandwidth_id = torch.tensor([0]).to(features_input.device)
        x = self.backbone(features_input, bandwidth_id=bandwidth_id, **kwargs)
        audio_output = self.head(x)
        return audio_output

    def codes_to_features(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Transforms an input sequence of discrete tokens (codes) into feature embeddings using the feature extractor's
        codebook weights.

        Args:
            codes (Tensor): The input tensor. Expected shape is (L) or (B, L),
                            where K is the number of codebooks, B is the batch size and L is the sequence length.

        Returns:
            Tensor: Features of shape (B, C, L), where B is the batch size, C denotes the feature dimension,
                    and L is the sequence length.
        """
        assert isinstance(
            self.feature_extractor, EncodecFeatures
        ), "Feature extractor should be an instance of EncodecFeatures"

        if codes.dim() == 1:
            codes = rearrange(codes, "l -> 1 1 l")
        if codes.dim() == 2:
            codes = rearrange(codes, "b l -> 1 b l")
        n_bins = self.feature_extractor.encodec.quantizer.bins
        offsets = torch.arange(0, n_bins * len(codes), n_bins, device=codes.device)
        embeddings_idxs = codes + offsets.view(-1, 1, 1)

        tmp = torch.cat(
            [vq.codebook for vq in self.feature_extractor.encodec.quantizer.vq.layers],
            dim=0,
        )
        embeddings_idxs = embeddings_idxs.to(tmp.device)
        features = torch.nn.functional.embedding(embeddings_idxs, tmp).sum(dim=0)
        features = features.transpose(1, 2)

        return features

    @property
    def codebook_size(self):
        return self.feature_extractor.codebook_size

    @property
    def downsample_rate(self):
        return self.feature_extractor.downsample_rate


if __name__ == "__main__":
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[WavTokenizer Test] device={device}")

    # --- 1. Build model ---
    print("[1] Building WavTokenizer model...")
    model = WavTokenizer(
        pretrained="checkpoints/WavTokenizer-large-unify-40token"
    ).to(device).eval()

    print(f"   Codebook size: {model.codebook_size}")
    print(f"   Downsample rate: {model.downsample_rate}")
    sample_rate = 24000

    # --- 2. Load test audio (same file as UniCodec test) ---
    audio_path = "data/motionhub/motionx/audio/music/Play_Flute_4_clip1.wav"
    print(f"\n[2] Loading audio: {audio_path}")

    torchaudio = _import_torchaudio()
    wav, sr = torchaudio.load(audio_path)

    # Resample to 24kHz if needed
    if sr != sample_rate:
        print(f"   Resampling {sr} -> {sample_rate}...")
        wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)

    # Mono, first 5 seconds
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav[:, : sample_rate * 5]
    audio = wav.to(device)
    print(f"   Audio shape: {audio.shape}, duration: {audio.shape[-1]/sample_rate:.2f}s")

    # --- 3. Encode ---
    print("\n[3] Encoding...")
    t0 = time.time()
    with torch.no_grad():
        features, codes = model.encode(audio)
    t_enc = time.time() - t0
    print(f"   Features shape: {features.shape}")
    print(f"   Codes shape: {codes.shape}")
    print(f"   Encode time: {t_enc:.3f}s")
    n_tokens = codes.shape[-1]
    duration = audio.shape[-1] / sample_rate
    print(f"   Actual TPS: {n_tokens/duration:.1f}")

    # --- 4. Decode from features ---
    print("\n[4] Decoding from features...")
    t0 = time.time()
    with torch.no_grad():
        audio_recon_feat = model.decode(features, is_idx=False)
    t_dec = time.time() - t0
    print(f"   Recon audio shape: {audio_recon_feat.shape}")
    print(f"   Decode time: {t_dec:.3f}s")

    # --- 5. Decode from codes ---
    print("\n[5] Decoding from codes (indices)...")
    with torch.no_grad():
        audio_recon_idx = model.decode(codes, is_idx=True)
    print(f"   Recon audio shape: {audio_recon_idx.shape}")

    # --- 6. Save ---
    out_dir = "outputs/wavtokenizer_test"
    os.makedirs(out_dir, exist_ok=True)

    input_path = os.path.join(out_dir, "input.wav")
    recon_feat_path = os.path.join(out_dir, "recon_from_features.wav")
    recon_idx_path = os.path.join(out_dir, "recon_from_codes.wav")

    save_audio(audio.cpu(), input_path, sample_rate=sample_rate)
    save_audio(audio_recon_feat.cpu(), recon_feat_path, sample_rate=sample_rate)
    save_audio(audio_recon_idx.cpu(), recon_idx_path, sample_rate=sample_rate)

    print(f"\n[6] Saved outputs to: {out_dir}")
    print(f"   - {input_path}")
    print(f"   - {recon_feat_path}")
    print(f"   - {recon_idx_path}")
    print("\n[Done]")
