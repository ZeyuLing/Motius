#!/usr/bin/env python3
"""Convert the authors' UniMuMo package into a complete Motius artifact."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import numpy as np
import torch

from motius.models.unimumo.bundle import (
    DEFAULT_UNIMUMO_CONFIG,
    UNIMUMO_SOURCE_REPOSITORY,
    UNIMUMO_SOURCE_REVISION,
    UniMuMoBundle,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", default="t5-base")
    return parser.parse_args()


def _required(state, key, target_shape=None):
    if key not in state:
        raise KeyError(f"Official UniMuMo checkpoint misses {key!r}")
    value = state[key]
    if target_shape is not None and value.shape != target_shape:
        raise ValueError(
            f"Shape mismatch for {key}: {tuple(value.shape)} != {tuple(target_shape)}"
        )
    return value


def _motion_source_key(target_key: str) -> str:
    if target_key.startswith("encoder.init_conv."):
        return "motion_encoder." + target_key[len("encoder.") :]
    if target_key.startswith("encoder.post_conv."):
        return "motion_encoder." + target_key[len("encoder.") :]
    match = re.match(r"encoder\.blocks\.(\d+)\.1\.(\d+)\.(.+)", target_key)
    if match:
        block, depth, suffix = match.groups()
        return f"motion_encoder.resnet_model.{block}.1.model.{depth}.{suffix}"
    match = re.match(r"encoder\.blocks\.(\d+)\.0\.(.+)", target_key)
    if match:
        block, suffix = match.groups()
        return f"motion_encoder.resnet_model.{block}.0.{suffix}"
    if target_key.startswith("decoder.init_conv."):
        return "motion_decoder." + target_key[len("decoder.") :]
    if target_key.startswith("decoder.post_conv."):
        return "motion_decoder." + target_key[len("decoder.") :]
    match = re.match(r"decoder\.blocks\.(\d+)\.0\.(\d+)\.(.+)", target_key)
    if match:
        block, depth, suffix = match.groups()
        return f"motion_decoder.resnet_block.{block}.0.model.{depth}.{suffix}"
    match = re.match(r"decoder\.blocks\.(\d+)\.1\.(.+)", target_key)
    if match:
        block, suffix = match.groups()
        return f"motion_decoder.resnet_block.{block}.1.{suffix}"
    if target_key.startswith("pre_quantize."):
        return "pre_quantize_conv." + target_key[len("pre_quantize.") :]
    if target_key.startswith("post_quantize."):
        return "post_quantize_conv." + target_key[len("post_quantize.") :]
    raise KeyError(f"No official motion-codec mapping for {target_key!r}")


def map_motion_codec(state, module):
    mapped = {}
    for key, target in module.state_dict().items():
        source = _motion_source_key(key)
        mapped[key] = _required(state, source, target.shape)
    return mapped


def _split_attention(mapped, target_state, prefix, source_state, source_prefix):
    weight = _required(
        source_state,
        source_prefix + ".in_proj_weight",
        torch.Size((target_state[prefix + ".q_proj.weight"].shape[0] * 3,) + target_state[prefix + ".q_proj.weight"].shape[1:]),
    )
    q_weight, k_weight, v_weight = weight.chunk(3, dim=0)
    mapped[prefix + ".q_proj.weight"] = q_weight
    mapped[prefix + ".k_proj.weight"] = k_weight
    mapped[prefix + ".v_proj.weight"] = v_weight
    mapped[prefix + ".out_proj.weight"] = _required(
        source_state,
        source_prefix + ".out_proj.weight",
        target_state[prefix + ".out_proj.weight"].shape,
    )
    bias_key = source_prefix + ".in_proj_bias"
    if prefix + ".q_proj.bias" in target_state:
        bias = _required(source_state, bias_key)
        q_bias, k_bias, v_bias = bias.chunk(3, dim=0)
        mapped[prefix + ".q_proj.bias"] = q_bias
        mapped[prefix + ".k_proj.bias"] = k_bias
        mapped[prefix + ".v_proj.bias"] = v_bias
        mapped[prefix + ".out_proj.bias"] = _required(
            source_state, source_prefix + ".out_proj.bias"
        )


def map_generator(state, module):
    target = module.state_dict()
    mapped = {}
    for index in range(module.num_codebooks):
        for destination, source in (
            (f"music_embeddings.{index}.weight", f"model.emb.{index}.weight"),
            (f"motion_embeddings.{index}.weight", f"model.motion_emb.{index}.weight"),
            (f"music_heads.{index}.weight", f"model.linears.{index}.weight"),
            (f"motion_heads.{index}.weight", f"model.motion_linears.{index}.weight"),
        ):
            mapped[destination] = _required(state, source, target[destination].shape)

    layer_names = {
        "music_ffn_in.weight": "linear1.weight",
        "music_ffn_out.weight": "linear2.weight",
        "motion_ffn_in.weight": "linear1_motion.weight",
        "motion_ffn_out.weight": "linear2_motion.weight",
        "self_norm.weight": "norm1.weight",
        "self_norm.bias": "norm1.bias",
        "motion_self_norm.weight": "norm1_motion.weight",
        "motion_self_norm.bias": "norm1_motion.bias",
        "music_ffn_norm.weight": "norm2.weight",
        "music_ffn_norm.bias": "norm2.bias",
        "motion_ffn_norm.weight": "norm2_motion.weight",
        "motion_ffn_norm.bias": "norm2_motion.bias",
        "cross_norm.weight": "norm_cross.weight",
        "cross_norm.bias": "norm_cross.bias",
    }
    for index in range(len(module.layers)):
        destination_prefix = f"layers.{index}"
        source_prefix = f"model.transformer.layers.{index}"
        for destination, source in layer_names.items():
            destination = f"{destination_prefix}.{destination}"
            mapped[destination] = _required(
                state, f"{source_prefix}.{source}", target[destination].shape
            )
        for destination, source in (
            ("self_attention", "self_attn"),
            ("caption_attention", "captioning_self_attn"),
            ("cross_attention", "cross_attention"),
        ):
            _split_attention(
                mapped,
                target,
                f"{destination_prefix}.{destination}",
                state,
                f"{source_prefix}.{source}",
            )
    if "output_norm.weight" in target:
        mapped["output_norm.weight"] = _required(
            state, "model.out_norm.weight", target["output_norm.weight"].shape
        )
        mapped["output_norm.bias"] = _required(
            state, "model.out_norm.bias", target["output_norm.bias"].shape
        )
    missing = sorted(set(target) - set(mapped))
    if missing:
        raise KeyError(f"Generator mapping misses target keys: {missing}")
    return mapped


def _audio_source_key(target_key: str) -> str:
    if target_key.startswith("quantizer.layers."):
        return target_key.replace(
            "quantizer.layers.", "quantizer.vq.layers.", 1
        ).replace(".codebook.", "._codebook.")
    match = re.match(r"(encoder|decoder)\.layers\.(\d+)\.(.+)", target_key)
    if not match:
        raise KeyError(f"No official Encodec mapping for {target_key!r}")
    side, index_text, suffix = match.groups()
    index = int(index_text)
    prefix = f"{side}.model.{index}"
    if side == "decoder" and index in {3, 6, 9, 12}:
        suffix = suffix.replace("conv.parametrizations.weight.original0", "convtr.convtr.weight_g")
        suffix = suffix.replace("conv.parametrizations.weight.original1", "convtr.convtr.weight_v")
        suffix = suffix.replace("conv.bias", "convtr.convtr.bias")
    else:
        suffix = suffix.replace("conv.parametrizations.weight.original0", "conv.conv.weight_g")
        suffix = suffix.replace("conv.parametrizations.weight.original1", "conv.conv.weight_v")
        suffix = suffix.replace("conv.bias", "conv.conv.bias")
    return f"{prefix}.{suffix}"


def map_audio_codec(state, module):
    mapped = {}
    for key, target in module.state_dict().items():
        source = _audio_source_key(key)
        mapped[key] = _required(state, source, target.shape)
    return mapped


def prefixed_state(state, prefix, target_state):
    mapped = {}
    for key, target in target_state.items():
        mapped[key] = _required(state, prefix + key, target.shape)
    return mapped


def build_audio_codec():
    from transformers import EncodecConfig, EncodecModel

    config = EncodecConfig(
        audio_channels=1,
        chunk_length_s=None,
        codebook_dim=128,
        codebook_size=2048,
        compress=2,
        dilation_growth_rate=2,
        hidden_size=128,
        kernel_size=7,
        last_kernel_size=7,
        norm_type="weight_norm",
        normalize=False,
        num_filters=64,
        num_lstm_layers=2,
        num_residual_layers=1,
        overlap=None,
        pad_mode="reflect",
        residual_kernel_size=3,
        sampling_rate=32_000,
        target_bandwidths=[2.2],
        trim_right_ratio=1.0,
        upsampling_ratios=[8, 5, 4, 4],
        use_causal_conv=False,
        use_conv_shortcut=False,
    )
    return EncodecModel(config)


def main():
    args = parse_args()
    package = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    required = {
        "motion_mean",
        "motion_std",
        "music_vqvae_weight",
        "motion_vqvae_weight",
        "music_motion_lm_weight",
    }
    missing = sorted(required - set(package))
    if missing:
        raise KeyError(f"Official package misses fields: {missing}")

    from transformers import AutoTokenizer, T5Config, T5EncoderModel
    from transformers import T5ForConditionalGeneration

    text_config = T5Config.from_pretrained(args.tokenizer)
    audio_codec = build_audio_codec()
    text_encoder = T5EncoderModel(text_config)
    captioner = T5ForConditionalGeneration(copy.deepcopy(text_config))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_tokens("<separation>")
    bundle = UniMuMoBundle(
        DEFAULT_UNIMUMO_CONFIG,
        audio_codec=audio_codec,
        text_encoder=text_encoder,
        captioner=captioner,
        tokenizer=tokenizer,
        mean=np.asarray(package["motion_mean"], dtype=np.float32),
        std=np.asarray(package["motion_std"], dtype=np.float32),
        provenance={
            "source_repository": UNIMUMO_SOURCE_REPOSITORY,
            "source_revision": UNIMUMO_SOURCE_REVISION,
            "source_checkpoint": args.checkpoint.name,
        },
    )

    audio_report = bundle.audio_codec.load_state_dict(
        map_audio_codec(package["music_vqvae_weight"], bundle.audio_codec),
        strict=True,
    )
    motion_report = bundle.motion_codec.load_state_dict(
        map_motion_codec(package["motion_vqvae_weight"], bundle.motion_codec),
        strict=True,
    )
    generator_report = bundle.generator.load_state_dict(
        map_generator(package["music_motion_lm_weight"], bundle.generator),
        strict=True,
    )
    lm_state = package["music_motion_lm_weight"]
    condition_prefix = "model.condition_provider.conditioners.description.t5."
    caption_prefix = "text_model.model."
    text_report = bundle.text_encoder.load_state_dict(
        prefixed_state(lm_state, condition_prefix, bundle.text_encoder.state_dict()),
        strict=True,
    )
    caption_report = bundle.captioner.load_state_dict(
        prefixed_state(lm_state, caption_prefix, bundle.captioner.state_dict()),
        strict=True,
    )
    projection_state = {
        "text_condition_projection.weight": _required(
            lm_state,
            "model.condition_provider.conditioners.description.output_proj.weight",
            bundle.core.text_condition_projection.weight.shape,
        ),
        "text_condition_projection.bias": _required(
            lm_state,
            "model.condition_provider.conditioners.description.output_proj.bias",
            bundle.core.text_condition_projection.bias.shape,
        ),
        "caption_context_projection.weight": _required(
            lm_state,
            "text_model.context_proj.weight",
            bundle.core.caption_context_projection.weight.shape,
        ),
        "caption_context_projection.bias": _required(
            lm_state,
            "text_model.context_proj.bias",
            bundle.core.caption_context_projection.bias.shape,
        ),
    }
    projection_report = bundle.core.load_state_dict(projection_state, strict=False)
    expected_missing = {
        key
        for key in bundle.core.state_dict()
        if not key.startswith("text_condition_projection.")
        and not key.startswith("caption_context_projection.")
    }
    if set(projection_report.missing_keys) != expected_missing:
        raise RuntimeError("Unexpected projection loading report")
    reports = [
        audio_report,
        motion_report,
        generator_report,
        text_report,
        caption_report,
    ]
    if any(report.missing_keys or report.unexpected_keys for report in reports):
        raise RuntimeError(f"Strict component load failed: {reports}")

    bundle.save_pretrained(args.output)
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "artifact_format": "motius-unimumo-v1",
                "status": "strictly mapped",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
