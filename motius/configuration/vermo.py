import os
import torch


def export_processor_assets(config, directory):
    """Package declared tokenizer/statistics dependencies outside neural modules."""
    import copy
    import json
    import shutil
    from pathlib import Path

    config = copy.deepcopy(config)
    directory = Path(directory)
    processor = config["processor"]
    tokenizer = processor["pretrained_text_tokenizer"]["from_pretrained"]
    source = Path(tokenizer["pretrained_model_name_or_path"])
    target = directory / "tokenizer"
    if source.resolve() != target.resolve():
        target.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.is_file() and path.suffix in {".json", ".txt", ".model"}:
                shutil.copy2(path, target / path.name)
    tokenizer["pretrained_model_name_or_path"] = "tokenizer"
    for name in ["smpl_pose_processor", "multi_person_smpl_pose_processor"]:
        if processor.get(name) is None:
            continue
        source = Path(processor[name]["stats_file"])
        relative = "stats/" + name + ".json"
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        processor[name]["stats_file"] = relative
    (directory / "bundle_config.json").write_text(json.dumps(config, indent=2) + "\n")
    metadata = dict(
        _class_name="VermoPipeline", _library_name="motius", model_type="vermo",
        pipeline_class="motius.pipelines.vermo.VermoPipeline",
        bundle_class="motius.models.vermo.VermoBundle", tasks=["motion_to_text"],
    )
    (directory / "model_index.json").write_text(json.dumps(metadata, indent=2) + "\n")


def export_audio_network(bundle, config, directory):
    """Keep an optional current WavTokenizer together with its neural config."""
    import shutil
    from pathlib import Path

    if bundle.audio_tokenizer is None:
        return
    options = config["processor"]["audio_tokenizer"]
    source = Path(options["pretrained"])
    architecture = sorted(source.rglob("*.yaml"))
    if len(architecture) != 1:
        raise ValueError("WavTokenizer export needs one declared architecture YAML")
    target = Path(directory) / "audio_tokenizer"
    target.mkdir(parents=True, exist_ok=True)
    if architecture[0].resolve() != (target / "config.yaml").resolve():
        shutil.copy2(architecture[0], target / "config.yaml")
    network = getattr(bundle.audio_tokenizer, "module", bundle.audio_tokenizer)
    torch.save({"state_dict": network.state_dict()}, target / "model.ckpt")
    options["pretrained"] = "audio_tokenizer"
    config["processor"]["pretrained_audio_tokenizer"] = None

def _disable_flash_attention_packed_sequence_check() -> None:
    """Avoid a GPU-side packed-sequence probe for non-packed VerMo batches."""
    if os.environ.get('VERMO_DISABLE_FA2_PACKED_SEQUENCE_CHECK', '1') == '0':
        return
    try:
        import transformers.modeling_flash_attention_utils as fa_utils
        import transformers.models.llama.modeling_llama as llama_modeling
    except Exception:
        return

    def _not_packed_sequence(position_ids, batch_size):
        return False
    fa_utils._is_packed_sequence = _not_packed_sequence

    def _wrap_flash_attention_forward(fn):
        if getattr(fn, '_vermo_no_position_ids_probe', False):
            return fn

        def _flash_attention_forward_without_position_probe(*args, **kwargs):
            args = list(args)
            if len(args) >= 8:
                args[7] = None
                kwargs.pop('position_ids', None)
            else:
                kwargs['position_ids'] = None
            return fn(*args, **kwargs)
        _flash_attention_forward_without_position_probe._vermo_no_position_ids_probe = True
        return _flash_attention_forward_without_position_probe
    fa_utils._flash_attention_forward = _wrap_flash_attention_forward(fa_utils._flash_attention_forward)
    if hasattr(llama_modeling, '_flash_attention_forward'):
        llama_modeling._flash_attention_forward = fa_utils._flash_attention_forward

def _configure_sdp_backends() -> None:
    backend = os.environ.get('VERMO_SDP_BACKEND', '').strip().lower().replace('-', '_')
    if not backend or not hasattr(torch.backends, 'cuda'):
        return
    cuda_backends = torch.backends.cuda
    if backend in {'no_flash', 'mem_efficient', 'mem_efficient_or_math'}:
        cuda_backends.enable_flash_sdp(False)
        cuda_backends.enable_mem_efficient_sdp(True)
        cuda_backends.enable_math_sdp(True)
        if hasattr(cuda_backends, 'enable_cudnn_sdp'):
            cuda_backends.enable_cudnn_sdp(False)
    elif backend == 'math':
        cuda_backends.enable_flash_sdp(False)
        cuda_backends.enable_mem_efficient_sdp(False)
        if hasattr(cuda_backends, 'enable_cudnn_sdp'):
            cuda_backends.enable_cudnn_sdp(False)
        cuda_backends.enable_math_sdp(True)

def configure_attention_runtime():
    _disable_flash_attention_packed_sequence_check()
    _configure_sdp_backends()
