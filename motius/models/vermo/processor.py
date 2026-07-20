from collections import defaultdict
import os
import random
import time
from typing import Dict, List, Optional, Tuple, Union
from einops import rearrange

from tokenizers import AddedToken
from torch import Tensor
import torch
from torch import nn
import sys
import os
from mmengine.runner import load_checkpoint
from mmengine import print_log
from tqdm import tqdm
from motius.models.vermo.wavtokenizer import WavTokenizer
from motius.models.vermo.vqvae_2d import VQVAEVermo2DTK as VQVAEWanMotion2DTK
from motius.models.vermo.vqvae_1d import VQVAEVermo1D as VQVAEWanMotion1D
from motius.models.vermo.pose_processor import VermoSMPL22Processor as SMPLPoseProcessor


from motius.models.vermo.task_utils import (
    ALL_MODALS,
    LOCATABLE_MODALS,
)
from motius.models.vermo.task_utils.task_lib.base_task import (
    BaseTask,
)

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils_fast import BatchEncoding
from mmengine.config import Config
from motius.registry import HF_MODELS, MODELS
from motius.models.utils import print_colored_log
from motius.models.vermo.task_utils.modality import (
    Audio,
    Caption,
    Duration,
    FutureMotion,
    FutureMusic,
    Genre,
    MiddleMotion,
    Modality,
    Motion,
    Music,
    NumPerson,
    PastMotion,
    PastMusic,
    SpeechScript,
    Text,
    is_modal,
)


def obtain_data(
    modal: Modality, inputs: Dict, sample_idx: int, allow_none: bool = False
):
    key = modal.data_keys[0]
    if key not in inputs:
        if allow_none:
            return None
        else:
            raise ValueError(f"Key {key} not found in inputs")
    value = inputs[key][sample_idx]
    return value


def format_text_modal_data(modal: Modality, modal_data: Union[str, List[str], Tuple[str, ...]]) -> str:
    if isinstance(modal_data, (list, tuple)):
        texts = [str(x).strip() for x in modal_data if str(x).strip()]
        if is_modal(modal, Caption):
            return "\n".join(
                f"Person {idx + 1} caption: {text}"
                for idx, text in enumerate(texts)
            )
        return "\n".join(texts)
    return str(modal_data)


def pad_training_token_sequences(
    sequences: List[Tensor],
    pad_id: int,
    instruction_stage: bool,
    output_bos_positions: List[int],
) -> Tuple[Tensor, Optional[Tensor], Tensor]:
    """Right-pad causal-LM training batches without fully masked query rows."""
    if not sequences:
        raise ValueError("Expected at least one training token sequence")
    if len(sequences) != len(output_bos_positions):
        raise ValueError("Sequence and output BOS position counts must match")

    device = sequences[0].device
    seq_lengths = [sequence.numel() for sequence in sequences]
    max_len = max(seq_lengths)
    needs_padding = any(seq_len != max_len for seq_len in seq_lengths)
    input_ids = torch.full(
        (len(sequences), max_len), pad_id, dtype=torch.long, device=device
    )
    labels = torch.full_like(input_ids, -100)
    attention_mask = (
        torch.zeros_like(input_ids)
        if needs_padding else None
    )

    for sample_idx, sequence in enumerate(sequences):
        seq_len = sequence.numel()
        input_ids[sample_idx, :seq_len] = sequence
        labels[sample_idx, :seq_len] = sequence
        if attention_mask is not None:
            attention_mask[sample_idx, :seq_len] = 1
        if instruction_stage:
            labels[sample_idx, : output_bos_positions[sample_idx] + 1] = -100

    return input_ids, attention_mask, labels


@MODELS.register_module()
class VermoProcessor(nn.Module):

    def __init__(
        self,
        pretrained_text_tokenizer: Dict,
        smpl_pose_processor: Dict,
        motion_tokenizer: Dict,
        multi_person_smpl_pose_processor: Optional[Dict] = None,
        audio_tokenizer: Optional[Dict] = None,
        pretrained_audio_tokenizer: Optional[str] = None,
        audio_codebook_size: int = 4096,
        instruction_stage: bool = False,
        optional_input_modal_mode: str = "random",
        task_template_mode: str = "random",
        shuffle_modal_parts: bool = True,
        shuffle_condition_parts: Optional[bool] = None,
        shuffle_output_parts: Optional[bool] = None,
        max_seq_len: int = 0,
    ):
        """

        :param pretrained_text_tokenizer: path of the pretrained text tokenizer
        :param motion_tokenizer: config of the motion tokenizer, the "type" key can be a path if u don't want to copy the complex config file.
        We will use the "model" config in the config file to initialize the tokenizer with pretrained weights
        :param audio_tokenizer: config of the audio tokenizer.  Optional — when None the
            audio vocab is still reserved (using ``audio_codebook_size``) so the LLM
            vocabulary stays compatible, but the ~300 MB WavTokenizer weights are not loaded.
            Set to None for stages that don't involve audio tasks (e.g. stage 0).
        :param audio_codebook_size: codebook size used for audio vocab reservation when
            ``audio_tokenizer`` is None.  Must match the actual WavTokenizer codebook
            size (default 4096) to keep checkpoint-compatible vocabularies.
        :param optional_input_modal_mode: mode of optional input modal, defaults to "random".
        During training, we randomly select some optional input modalities to include in the input condition.
        During evalluation, if the mode is "none", we will not include any optional input modalities in the input condition.
        if the mode is "all", we will include all optional input modalities in the input condition.
        if the mode is "duration", we will only include the duration modality in the input condition(if duration is available in the task).
        if the mode is "caption", we will only include the caption modality in the input condition(if caption is available in the task).
        :param max_seq_len: maximum token sequence length for training.  Sequences
            longer than this are truncated from the right (output tokens are clipped).
            0 means no limit.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.smpl_pose_processor: SMPLPoseProcessor = MODELS.build(smpl_pose_processor)
        self.multi_person_smpl_pose_processor: Optional[SMPLPoseProcessor] = (
            MODELS.build(multi_person_smpl_pose_processor)
            if multi_person_smpl_pose_processor is not None
            else None
        )

        self.text_tokenizer: PreTrainedTokenizer = HF_MODELS.build(
            pretrained_text_tokenizer
        )
        self.text_tokenizer.padding_side = "left"
        self.motion_tokenizer: VQVAEWanMotion2DTK = HF_MODELS.build(motion_tokenizer)

        if audio_tokenizer is not None:
            self.audio_tokenizer: Optional[WavTokenizer] = MODELS.build(audio_tokenizer)
            if pretrained_audio_tokenizer is not None:
                load_checkpoint(
                    self.audio_tokenizer, pretrained_audio_tokenizer, map_location="cpu"
                )
                print_log(f"Load audio tokenizer from {pretrained_audio_tokenizer}")
            self._audio_codebook_size = self.audio_tokenizer.codebook_size
        else:
            self.audio_tokenizer = None
            self._audio_codebook_size = audio_codebook_size
            print_log(
                f"Audio tokenizer skipped; reserving {audio_codebook_size} audio tokens "
                f"for vocab compatibility."
            )

        self.instruction_stage = instruction_stage
        self.optional_input_modal_mode = optional_input_modal_mode
        self.task_template_mode = task_template_mode
        self.shuffle_modal_parts = bool(shuffle_modal_parts)
        self.shuffle_condition_parts = (
            self.shuffle_modal_parts
            if shuffle_condition_parts is None
            else bool(shuffle_condition_parts)
        )
        self.shuffle_output_parts = (
            self.shuffle_modal_parts
            if shuffle_output_parts is None
            else bool(shuffle_output_parts)
        )
        assert self.optional_input_modal_mode in [
            "none",
            "all",
            "duration",
            "caption",
            "random",
        ]
        assert self.task_template_mode in ["random", "first"]
        self.set_vocab()

    def _runtime_device(self) -> torch.device:
        for module in (
            self.motion_tokenizer,
            self.smpl_pose_processor,
            self.multi_person_smpl_pose_processor,
            self.audio_tokenizer,
        ):
            if isinstance(module, nn.Module):
                try:
                    return next(module.parameters()).device
                except StopIteration:
                    continue
        return torch.device('cpu')

    def _smpl_processor_for_num_person(self, num_person: int) -> SMPLPoseProcessor:
        if num_person > 1 and self.multi_person_smpl_pose_processor is not None:
            return self.multi_person_smpl_pose_processor
        return self.smpl_pose_processor

    def _debug_process_phases(self) -> bool:
        return os.environ.get('VERMO_DEBUG_PROCESS_PHASES') == '1'

    def _debug_rank(self) -> int:
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return int(torch.distributed.get_rank())
        except Exception:
            pass
        return int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))

    def _debug_phase_log(self, message: str) -> None:
        if self._debug_process_phases():
            print(f'[VerMo process phase rank={self._debug_rank()}] {message}', flush=True)

    def _debug_cuda_sync(self) -> None:
        if (
            self._debug_process_phases()
            and os.environ.get('VERMO_DEBUG_CUDA_SYNC') == '1'
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize()

    def _h2d_copy_lock_enabled(self) -> bool:
        return os.environ.get("VERMO_H2D_COPY_LOCK", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _h2d_copy_sync_enabled(self) -> bool:
        return os.environ.get("VERMO_H2D_COPY_SYNC", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _pin_motion_batch_enabled(self) -> bool:
        return os.environ.get("VERMO_PIN_MOTION_BATCH", "1").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _bind_runtime_cuda_device(self, device: torch.device) -> None:
        if device.type != "cuda" or not torch.cuda.is_available():
            return
        if device.index is not None:
            torch.cuda.set_device(device.index)

    def _copy_tensor_to_runtime_device(self, tensor: Tensor, device: torch.device) -> Tensor:
        self._bind_runtime_cuda_device(device)
        non_blocking = bool(getattr(tensor, "is_pinned", lambda: False)())
        if tensor.device == device:
            return tensor
        if tensor.device.type != "cpu" or not self._h2d_copy_lock_enabled():
            return tensor.to(device, non_blocking=non_blocking)

        lock_dir = os.environ.get("VERMO_H2D_COPY_LOCK_DIR", "/tmp")
        lock_path = os.path.join(lock_dir, "vermo_h2d_copy.lock")
        try:
            import fcntl

            os.makedirs(lock_dir, exist_ok=True)
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                self._bind_runtime_cuda_device(device)
                copied = tensor.to(device, non_blocking=non_blocking)
                if (
                    self._h2d_copy_sync_enabled()
                    and device.type == "cuda"
                    and torch.cuda.is_available()
                ):
                    torch.cuda.synchronize(device)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            return copied
        except Exception as exc:
            self._debug_phase_log(f"h2d copy lock fallback: {exc}")
            return tensor.to(device, non_blocking=non_blocking)

    def _ids_to_runtime_device(self, ids: List[int], device: Optional[torch.device] = None) -> Tensor:
        if device is None:
            device = self._runtime_device()
        pin_ids = (
            self._pin_motion_batch_enabled()
            and device.type == "cuda"
            and torch.cuda.is_available()
        )
        try:
            cpu_ids = torch.empty((len(ids),), dtype=torch.long, pin_memory=pin_ids)
            if ids:
                cpu_ids.copy_(torch.tensor(ids, dtype=torch.long))
        except Exception as exc:
            self._debug_phase_log(f"pinned id tensor fallback: {exc}")
            cpu_ids = torch.tensor(ids, dtype=torch.long)
        return self._copy_tensor_to_runtime_device(cpu_ids, device)

    def _motion_to_runtime_device(self, motion: Tensor) -> Tensor:
        device = self._runtime_device()
        if motion.device.type == "cpu" and motion.dtype != torch.float32:
            motion = motion.float()
        return self._copy_tensor_to_runtime_device(motion, device).float()

    def _motion_indices_to_cpu(self, indices):
        if isinstance(indices, list):
            return [self._motion_indices_to_cpu(item) for item in indices]
        if indices.device.type == "cpu" and indices.dtype == torch.long:
            return indices
        return indices.detach().to("cpu", dtype=torch.long)

    def _motion_tokenizer_cudnn_enabled(self) -> bool:
        return os.environ.get("VERMO_DISABLE_MOTION_TOKENIZER_CUDNN", "0") != "1"

    def _motion_tokenizer_cudnn_benchmark(self) -> bool:
        return os.environ.get("VERMO_MOTION_TOKENIZER_CUDNN_BENCHMARK", "0") == "1"

    def _motion_tokenizer_cudnn_deterministic(self) -> bool:
        return os.environ.get("VERMO_MOTION_TOKENIZER_CUDNN_DETERMINISTIC", "0") == "1"

    def _motion_tokenizer_cudnn_flags(self) -> Dict[str, bool]:
        return dict(
            enabled=self._motion_tokenizer_cudnn_enabled(),
            benchmark=self._motion_tokenizer_cudnn_benchmark(),
            deterministic=self._motion_tokenizer_cudnn_deterministic(),
            allow_tf32=True,
        )

    def set_vocab(self):
        self.ori_vocab_size = len(self.text_tokenizer.get_vocab())
        print_colored_log(f"[VOCAB DEBUG] Initial vocab size: {self.ori_vocab_size}")

        pad_token = "<|left_padding|>"
        self.text_tokenizer.add_tokens(
            AddedToken(
                pad_token,
                lstrip=False,
                rstrip=False,
                special=True,
            ),
            special_tokens=True,
        )
        self.text_tokenizer.pad_token = pad_token
        self.text_tokenizer.pad_token_id = self.text_tokenizer.convert_tokens_to_ids(
            pad_token
        )

        if self.text_tokenizer.bos_token is None:
            self.text_tokenizer.bos_token = "<|begin_of_text|>"
            self.text_tokenizer.add_tokens(
                AddedToken(
                    self.text_tokenizer.bos_token,
                    lstrip=False,
                    rstrip=False,
                    special=True,
                ),
                special_tokens=True,
            )
            print_colored_log(f"[VOCAB DEBUG] Added bos_token, total={len(self.text_tokenizer.get_vocab())}")
        else:
            print_colored_log(f"[VOCAB DEBUG] bos_token already exists: {self.text_tokenizer.bos_token}, total={len(self.text_tokenizer.get_vocab())}")

        print_colored_log(
            f"pad_token: {self.text_tokenizer.pad_token} - {self.text_tokenizer.pad_token_id},"
            f" bos_token: {self.text_tokenizer.bos_token} - {self.text_tokenizer.bos_token_id},"
            f" eos_token: {self.text_tokenizer.eos_token} - {self.text_tokenizer.eos_token_id}"
        )

        # 自定义 chat template，去掉 LLaMA-3 默认自动添加的 system message
        # 原始 LLaMA-3 模板会自动插入 "Cutting Knowledge Date..." 等内容
        custom_chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "{{ message['content'] }}<|eot_id|>"
            "{% elif message['role'] == 'assistant' %}"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "{{ message['content'] }}<|eot_id|>"
            "{% endif %}"
            "{% endfor %}"
        )
        self.text_tokenizer.chat_template = custom_chat_template
        print_colored_log("Set custom chat template (removed auto system message)")

        # base special tokens
        # bos and eos for condition
        self.cond_bos = "<|begin_of_condition|>"
        n = self.text_tokenizer.add_tokens(
            AddedToken(
                self.cond_bos,
                lstrip=False,
                rstrip=False,
                special=True,
            ),
            special_tokens=True,
        )
        print_colored_log(f"[VOCAB DEBUG] cond_bos: +{n}, total={len(self.text_tokenizer.get_vocab())}")
        self.cond_eos = "<|end_of_condition|>"
        n = self.text_tokenizer.add_tokens(
            AddedToken(
                self.cond_eos,
                lstrip=False,
                rstrip=False,
                special=True,
            ),
            special_tokens=True,
        )
        print_colored_log(f"[VOCAB DEBUG] cond_eos: +{n}, total={len(self.text_tokenizer.get_vocab())}")
        print_colored_log(
            f"BOS of condition, {self.cond_bos}: {self.text_tokenizer.convert_tokens_to_ids(self.cond_bos)}"
        )
        print_colored_log(
            f"EOS of condition, {self.cond_eos}: {self.text_tokenizer.convert_tokens_to_ids(self.cond_eos)}"
        )

        # bos and eos for task template
        self.task_bos = "<|begin_of_task_template|>"
        n = self.text_tokenizer.add_tokens(
            AddedToken(
                self.task_bos,
                lstrip=False,
                rstrip=False,
                special=True,
            ),
            special_tokens=True,
        )
        print_colored_log(f"[VOCAB DEBUG] task_bos: +{n}, total={len(self.text_tokenizer.get_vocab())}")
        self.task_eos = "<|end_of_task_template|>"
        n = self.text_tokenizer.add_tokens(
            AddedToken(
                self.task_eos,
                lstrip=False,
                rstrip=False,
                special=True,
            ),
            special_tokens=True,
        )
        print_colored_log(f"[VOCAB DEBUG] task_eos: +{n}, total={len(self.text_tokenizer.get_vocab())}")
        print_colored_log(
            f"BOS of task template, {self.task_bos}: {self.text_tokenizer.convert_tokens_to_ids(self.task_bos)}"
        )
        print_colored_log(
            f"EOS of task template, {self.task_eos}: {self.text_tokenizer.convert_tokens_to_ids(self.task_eos)}"
        )

        # bos and eos for output
        self.output_bos = "<|begin_of_output|>"
        n = self.text_tokenizer.add_tokens(
            AddedToken(
                self.output_bos,
                lstrip=False,
                rstrip=False,
                special=True,
            ),
            special_tokens=True,
        )
        print_colored_log(f"[VOCAB DEBUG] output_bos: +{n}, total={len(self.text_tokenizer.get_vocab())}")
        self.output_bos_id = self.text_tokenizer.convert_tokens_to_ids(self.output_bos)

        print_colored_log(f"BOS of output, {self.output_bos}: {self.output_bos_id}")
        print_colored_log(f"[VOCAB DEBUG] Before modal bos/eos loop: total={len(self.text_tokenizer.get_vocab())}")

        for modality in ALL_MODALS:
            if modality.bos is not None:
                bos = AddedToken(
                    modality.bos,
                    lstrip=False,
                    rstrip=False,
                    special=True,
                )
                n = self.text_tokenizer.add_tokens(bos, special_tokens=True)
                print_colored_log(
                    f"BOS of {modality.name}, {bos.content}: {self.text_tokenizer.convert_tokens_to_ids(bos.content)}, newly_added={n}"
                )
            if modality.eos is not None:
                eos = AddedToken(
                    modality.eos,
                    lstrip=False,
                    rstrip=False,
                    special=True,
                )
                n = self.text_tokenizer.add_tokens(eos, special_tokens=True)
                print_colored_log(
                    f"EOS of {modality.name}, {eos.content}: {self.text_tokenizer.convert_tokens_to_ids(eos.content)}, newly_added={n}"
                )
        # add the separator token of multi-person motion
        self.mp_separator = Motion.mp_separator
        n = self.text_tokenizer.add_tokens(
            AddedToken(
                self.mp_separator,
                lstrip=False,
                rstrip=False,
                special=True,
            ),
            special_tokens=True,
        )
        print_colored_log(
            f"Multi-person motion separator, {self.mp_separator}: {self.text_tokenizer.convert_tokens_to_ids(self.mp_separator)}, newly_added={n}"
        )
        print_colored_log(f"[VOCAB DEBUG] After modal bos/eos + separator: total={len(self.text_tokenizer.get_vocab())}")

        # extend the motion tokenizer and audio tokenizer to text_tokenizer vocab
        codebook_size = self.motion_tokenizer.codebook_size
        if hasattr(self.motion_tokenizer.quantizer, 'num_quantizers'):
            self._motion_num_quantizers = self.motion_tokenizer.quantizer.num_quantizers
            motion_vocab_size = codebook_size * self._motion_num_quantizers
        else:
            self._motion_num_quantizers = 1
            motion_vocab_size = codebook_size

        # Detect whether the motion tokenizer uses temporal downsampling.
        # 2D VQ-VAE always downsamples; 1D VQ-VAE only when config says so.
        if isinstance(self.motion_tokenizer, VQVAEWanMotion1D):
            td = list(self.motion_tokenizer.config.temporal_downsample)
            self._motion_has_temporal_ds = any(td)
        else:
            # 2D VQ-VAE always has WAN-style temporal downsampling
            self._motion_has_temporal_ds = True
        audio_vocab_size = self._audio_codebook_size

        self.add_modal_tokens(Motion, motion_vocab_size)
        print_colored_log(f"[VOCAB DEBUG] After Motion tokens ({motion_vocab_size}): total={len(self.text_tokenizer.get_vocab())}")
        self.add_modal_tokens(Audio, audio_vocab_size)
        print_colored_log(f"[VOCAB DEBUG] After Audio tokens ({audio_vocab_size}): total={len(self.text_tokenizer.get_vocab())}")
        print_colored_log(
            f"Original vocabulary size: {self.ori_vocab_size},"
            f" expanded vocabulary size: {len(self.text_tokenizer.get_vocab())}"
        )
        _vs_before_lut = len(self.text_tokenizer.get_vocab())
        self._build_modal_token_lut()
        _vs_after_lut = len(self.text_tokenizer.get_vocab())
        if _vs_after_lut != _vs_before_lut:
            print_colored_log(f"[VOCAB DEBUG] WARNING: _build_modal_token_lut changed vocab from {_vs_before_lut} to {_vs_after_lut}!")
        else:
            print_colored_log(f"[VOCAB DEBUG] _build_modal_token_lut did NOT change vocab, still {_vs_after_lut}")
        print_colored_log(f"[VOCAB DEBUG] Final vocab_size property = {self.vocab_size}")

    @property
    def vocab_size(self):
        return len(self.text_tokenizer.get_vocab())

    def add_modal_tokens(
        self,
        modal: Modality,
        num_tokens: int = 512,
    ) -> int:
        """
        高效地向 tokenizer 批量添加模态标记，并可选地同步扩展模型 embedding。

        - as_special=True 时：使用 additional_special_tokens，不会被分词器再切分，解码时可跳过
        - as_special=False 时：作为普通词添加

        返回：实际新增的 token 数（已存在的会自动去重）
        """
        texts: List[str] = [modal.token_format.format(i) for i in range(num_tokens)]

        n_added = self.text_tokenizer.add_tokens(
            texts, special_tokens=True
        )  # 支持 List[str] 或 List[AddedToken]

        print_colored_log(f"Add {n_added} tokens for {modal.name}")
        return n_added

    # ------------------------------------------------------------------
    # Lookup tables & batched encoding (training fast-path)
    # ------------------------------------------------------------------

    def _set_runtime_buffer(self, name: str, tensor: Tensor) -> None:
        if name in self._buffers:
            self._buffers[name] = tensor
        else:
            self.register_buffer(name, tensor, persistent=False)

    def _buffer_on_device(self, tensor: Tensor, device: torch.device) -> Tensor:
        device = torch.device(device)
        return self._copy_tensor_to_runtime_device(tensor, device)

    def _modal_bos_tensor(self, modal_name: str, device: torch.device) -> Tensor:
        name = self._modal_bos_buffer_names[modal_name]
        return self._buffer_on_device(getattr(self, name), device)

    def _modal_eos_tensor(self, modal_name: str, device: torch.device) -> Tensor:
        name = self._modal_eos_buffer_names[modal_name]
        return self._buffer_on_device(getattr(self, name), device)

    def _build_modal_token_lut(self):
        """Build VQ codebook index → LLM token ID lookup tables.

        Called once at the end of ``set_vocab()``.
        """
        tokenizer = self.text_tokenizer

        # Motion VQ index → LLM token ID
        motion_cb = self.motion_tokenizer.codebook_size
        total_motion_vocab = motion_cb * self._motion_num_quantizers
        self._set_runtime_buffer("motion_vq_to_lm", torch.tensor(
            [tokenizer.convert_tokens_to_ids(Motion.token_format.format(i))
             for i in range(total_motion_vocab)],
            dtype=torch.long,
        ))

        # Audio VQ index → LLM token ID
        audio_cb = self._audio_codebook_size
        self._set_runtime_buffer("audio_vq_to_lm", torch.tensor(
            [tokenizer.convert_tokens_to_ids(Audio.token_format.format(i))
             for i in range(audio_cb)],
            dtype=torch.long,
        ))

        # Pre-tokenize modal bos/eos as GPU tensors for fast cat
        self._modal_bos_ids: Dict[str, List[int]] = {}
        self._modal_eos_ids: Dict[str, List[int]] = {}
        self._modal_bos_t: Dict[str, Tensor] = {}
        self._modal_eos_t: Dict[str, Tensor] = {}
        self._modal_bos_buffer_names: Dict[str, str] = {}
        self._modal_eos_buffer_names: Dict[str, str] = {}
        for idx, modal in enumerate(ALL_MODALS):
            bos = tokenizer.encode(modal.bos, add_special_tokens=False) if modal.bos else []
            eos = tokenizer.encode(modal.eos, add_special_tokens=False) if modal.eos else []
            self._modal_bos_ids[modal.name] = bos
            self._modal_eos_ids[modal.name] = eos
            bos_name = f"_modal_bos_runtime_{idx}"
            eos_name = f"_modal_eos_runtime_{idx}"
            self._set_runtime_buffer(bos_name, torch.tensor(bos, dtype=torch.long))
            self._set_runtime_buffer(eos_name, torch.tensor(eos, dtype=torch.long))
            self._modal_bos_buffer_names[modal.name] = bos_name
            self._modal_eos_buffer_names[modal.name] = eos_name
            self._modal_bos_t[modal.name] = getattr(self, bos_name)
            self._modal_eos_t[modal.name] = getattr(self, eos_name)

        self._mp_separator_ids: List[int] = tokenizer.encode(
            Motion.mp_separator, add_special_tokens=False
        )
        self._set_runtime_buffer(
            "_mp_separator_t", torch.tensor(self._mp_separator_ids, dtype=torch.long)
        )

        # Chat template structural tokens (both list and tensor forms)
        self._chat_user_prefix_ids: List[int] = tokenizer.encode(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
            add_special_tokens=False,
        )
        self._chat_user_suffix_ids: List[int] = tokenizer.encode(
            "<|eot_id|>", add_special_tokens=False,
        )
        self._chat_asst_prefix_ids: List[int] = tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False,
        )
        self._chat_asst_suffix_ids: List[int] = tokenizer.encode(
            "<|eot_id|>", add_special_tokens=False,
        )
        self._task_bos_ids: List[int] = tokenizer.encode(
            self.task_bos, add_special_tokens=False,
        )
        self._task_eos_ids: List[int] = tokenizer.encode(
            self.task_eos, add_special_tokens=False,
        )
        self._cond_bos_ids: List[int] = tokenizer.encode(
            self.cond_bos, add_special_tokens=False,
        )
        self._cond_eos_ids: List[int] = tokenizer.encode(
            self.cond_eos, add_special_tokens=False,
        )
        self._output_bos_ids: List[int] = tokenizer.encode(
            self.output_bos, add_special_tokens=False,
        )

        # GPU tensor versions of structural tokens
        self._set_runtime_buffer(
            "_chat_user_prefix_t", torch.tensor(self._chat_user_prefix_ids, dtype=torch.long)
        )
        self._set_runtime_buffer(
            "_chat_user_suffix_t", torch.tensor(self._chat_user_suffix_ids, dtype=torch.long)
        )
        self._set_runtime_buffer(
            "_chat_asst_prefix_t", torch.tensor(self._chat_asst_prefix_ids, dtype=torch.long)
        )
        self._set_runtime_buffer(
            "_chat_asst_suffix_t", torch.tensor(self._chat_asst_suffix_ids, dtype=torch.long)
        )
        self._set_runtime_buffer("_task_bos_t", torch.tensor(self._task_bos_ids, dtype=torch.long))
        self._set_runtime_buffer("_task_eos_t", torch.tensor(self._task_eos_ids, dtype=torch.long))
        self._set_runtime_buffer("_cond_bos_t", torch.tensor(self._cond_bos_ids, dtype=torch.long))
        self._set_runtime_buffer("_cond_eos_t", torch.tensor(self._cond_eos_ids, dtype=torch.long))
        self._set_runtime_buffer(
            "_output_bos_t", torch.tensor(self._output_bos_ids, dtype=torch.long)
        )

        print_colored_log("Built modal VQ→LM lookup tables")

    @torch.no_grad()
    @torch.amp.autocast("cuda", enabled=False)
    def _encode_single_motion(
        self,
        motion: Tensor,
        smpl_pose_processor: Optional[SMPLPoseProcessor] = None,
    ) -> Tensor:
        """Encode a single motion tensor (no padding) through preprocessing + VQ-VAE.

        Args:
            motion: ``[P, T, C]`` tensor (already unsqueezed if single-person).

        Returns:
            ``LongTensor`` of VQ indices: ``[P, N]`` where N = T_out * K.
        """
        debug_t0 = time.perf_counter()
        debug_shape = tuple(motion.shape)
        self._debug_phase_log(
            f'encode_single start shape={debug_shape} device={motion.device}'
        )
        smpl_pose_processor = smpl_pose_processor or self.smpl_pose_processor
        # Preprocessing
        motion = self._motion_to_runtime_device(motion)
        self._debug_cuda_sync()
        self._debug_phase_log(
            f'encode_single to_device done elapsed={time.perf_counter() - debug_t0:.3f}s '
            f'shape={tuple(motion.shape)} device={motion.device}'
        )
        transl, pose = motion[..., :6], motion[..., 6:]
        transl = smpl_pose_processor.inv_convert_transl(transl)
        self._debug_cuda_sync()
        self._debug_phase_log(
            f'encode_single inv_convert_transl done elapsed={time.perf_counter() - debug_t0:.3f}s'
        )

        if self.motion_tokenizer.config.use_static:
            joints = smpl_pose_processor.fk(transl, pose)
            self._debug_cuda_sync()
            self._debug_phase_log(
                f'encode_single fk done elapsed={time.perf_counter() - debug_t0:.3f}s '
                f'joints_shape={tuple(joints.shape)}'
            )
            static_joints = smpl_pose_processor.get_static_joint_mask(
                joints[..., [7, 10, 8, 11, 20, 21], :],
                vel_thr=0.15,
                repeat_last=True,
            )
            self._debug_cuda_sync()
            self._debug_phase_log(
                f'encode_single static_mask done elapsed={time.perf_counter() - debug_t0:.3f}s '
                f'static_shape={tuple(static_joints.shape)}'
            )

        motion = smpl_pose_processor.normalize(motion)
        self._debug_cuda_sync()
        self._debug_phase_log(
            f'encode_single normalize done elapsed={time.perf_counter() - debug_t0:.3f}s '
            f'shape={tuple(motion.shape)}'
        )
        if self.motion_tokenizer.config.use_static:
            motion = torch.cat([motion, static_joints], dim=-1)
            self._debug_cuda_sync()
            self._debug_phase_log(
                f'encode_single cat_static done elapsed={time.perf_counter() - debug_t0:.3f}s '
                f'shape={tuple(motion.shape)}'
            )

        if isinstance(self.motion_tokenizer, VQVAEWanMotion1D):
            # 1D VQ-VAE: input [B, T, C], no reshape needed
            self._debug_phase_log(
                f'encode_single tokenizer_encode_1d start elapsed={time.perf_counter() - debug_t0:.3f}s'
            )
            with torch.backends.cudnn.flags(**self._motion_tokenizer_cudnn_flags()):
                indices = self.motion_tokenizer.encode(motion).indices
            self._debug_cuda_sync()
            self._debug_phase_log(
                f'encode_single tokenizer_encode_1d done elapsed={time.perf_counter() - debug_t0:.3f}s '
                f'indices_shape={tuple(indices.shape)}'
            )
            # RVQ: indices [B, N, num_q] → offset each layer and interleave → [B, N*num_q]
            if indices.ndim == 3:
                B, N, Q = indices.shape
                cb = self.motion_tokenizer.codebook_size
                offsets = torch.arange(Q, device=indices.device) * cb  # [Q]
                indices = indices + offsets  # broadcast → [B, N, Q]
                indices = indices.reshape(B, N * Q)  # interleaved
        else:
            # 2D VQ-VAE: input [B, T, K, C]
            motion = rearrange(motion, "b t (j d) -> b t j d", d=6)
            self._debug_cuda_sync()
            self._debug_phase_log(
                f'encode_single rearrange_2d done elapsed={time.perf_counter() - debug_t0:.3f}s '
                f'shape={tuple(motion.shape)}'
            )
            self._debug_phase_log(
                f'encode_single tokenizer_encode_2d start elapsed={time.perf_counter() - debug_t0:.3f}s'
            )
            with torch.backends.cudnn.flags(**self._motion_tokenizer_cudnn_flags()):
                indices = self.motion_tokenizer.encode(motion, flatten=True).indices
            self._debug_cuda_sync()
            self._debug_phase_log(
                f'encode_single tokenizer_encode_2d done elapsed={time.perf_counter() - debug_t0:.3f}s '
                f'indices_shape={tuple(indices.shape)}'
            )
        # indices: [P, N] where N = T_out * K
        return indices

    @torch.no_grad()
    def batch_encode_motion(
        self,
        motions: List[Tensor],
        per_person_num_frames_list: Optional[List[Optional[List[int]]]] = None,
    ) -> List[Tensor]:
        """Encode a list of motion tensors through preprocessing + VQ-VAE.

        Motions are grouped by person-count, padded to the same temporal
        length within each group, and encoded in a single batched forward
        pass per group.  Because both the 1-D and 2-D VQ-VAEs use **causal**
        convolutions, right-padding (replicate-last-frame) does **not**
        affect the indices of real (non-padded) frames — padded indices
        are simply trimmed after encoding.

        Args:
            motions: Each element is ``[T_i, C]`` (single-person) or ``[P, T_i, C]``
                (multi-person).
            per_person_num_frames_list: Optional list (same length as *motions*).
                Each element is either ``None`` (all persons share the same
                length, i.e. the tensor's T dimension) or a list of ints giving
                the real frame count **per person**.  When provided for
                multi-person motions, each person's indices are trimmed
                independently so that padding tokens from shorter persons are
                excluded.

        Returns:
            List per input (same order as *motions*):
            - Single-person: ``LongTensor [N_i]``
            - Multi-person without per-person lengths: ``LongTensor [P, N_i]``
            - Multi-person with per-person lengths: ``List[LongTensor]`` of P
              tensors, each ``[N_p]`` (possibly different lengths).
        """
        if len(motions) == 0:
            return []

        if per_person_num_frames_list is None:
            per_person_num_frames_list = [None] * len(motions)

        results: List = [None] * len(motions)

        # ── Step 1: Classify and record metadata ──────────────────────
        # group_key → list of (original_idx, motion_3d)
        # group_key = num_persons (1 for single-person, P for multi-person)
        groups: Dict[int, List[Tuple[int, Tensor, int]]] = defaultdict(list)
        for orig_idx, motion in enumerate(motions):
            if motion.ndim == 2:
                P = 1
                T = motion.shape[0]
                if T == 0:
                    results[orig_idx] = torch.zeros(
                        0, dtype=torch.long, device=self._runtime_device()
                    )
                    continue
                motion_3d = motion.unsqueeze(0)
            elif motion.ndim == 3:
                P = motion.shape[0]
                T = motion.shape[1]
                if T == 0:
                    results[orig_idx] = torch.zeros(
                        0, dtype=torch.long, device=self._runtime_device()
                    )
                    continue
                motion_3d = motion
            else:
                raise ValueError(f"Expected 2D or 3D motion tensor, got {motion.ndim}D")
            groups[P].append((orig_idx, motion_3d, T))

        # ── Step 2: Batched encode per group ──────────────────────────
        for P, items in groups.items():
            T_max = max(T for _, _, T in items)
            T_min = min(T for _, _, T in items)
            C = items[0][1].shape[-1]
            device = items[0][1].device
            runtime_device = self._runtime_device()
            pin_batch = (
                self._pin_motion_batch_enabled()
                and device.type == "cpu"
                and runtime_device.type == "cuda"
                and torch.cuda.is_available()
            )
            group_t0 = time.perf_counter()
            self._debug_phase_log(
                f'batch_encode group start P={P} batch={len(items)} '
                f'T_min={T_min} T_max={T_max} C={C} src_device={device} '
                f'runtime_device={runtime_device} pin_batch={pin_batch}'
            )

            # Pad all motions to T_max by replicating the last frame
            batch_shape = (len(items), P, T_max, C)
            batch_dtype = torch.float32 if pin_batch else items[0][1].dtype
            try:
                if pin_batch:
                    batch = torch.empty(batch_shape, dtype=batch_dtype, pin_memory=True)
                    batch.zero_()
                else:
                    batch = torch.zeros(batch_shape, device=device, dtype=batch_dtype)
            except RuntimeError as exc:
                self._debug_phase_log(f'pinned motion batch fallback: {exc}')
                pin_batch = False
                batch_dtype = items[0][1].dtype
                batch = torch.zeros(batch_shape, device=device, dtype=batch_dtype)
            real_lengths: List[int] = []
            orig_indices: List[int] = []
            single_person_flags: List[bool] = []

            for local_idx, (orig_idx, m, T) in enumerate(items):
                batch[local_idx, :, :T, :] = m
                if T < T_max:
                    # Replicate last frame for causal padding
                    batch[local_idx, :, T:, :] = m[:, T - 1 : T, :]
                real_lengths.append(T)
                orig_indices.append(orig_idx)
                single_person_flags.append(motions[orig_idx].ndim == 2)

            # Reshape [B, P, T, C] -> [B*P, T, C] for VQ-VAE
            batch_flat = batch.reshape(len(items) * P, T_max, C)
            self._debug_phase_log(
                f'batch_encode group padded P={P} batch_flat_shape={tuple(batch_flat.shape)} '
                f'elapsed={time.perf_counter() - group_t0:.3f}s'
            )

            # Preprocessing (vectorized over entire batch)
            smpl_pose_processor = self._smpl_processor_for_num_person(P)
            indices_flat = self._encode_single_motion(
                batch_flat,
                smpl_pose_processor=smpl_pose_processor,
            )
            self._debug_cuda_sync()
            self._debug_phase_log(
                f'batch_encode group encode done P={P} indices_shape={tuple(indices_flat.shape)} '
                f'elapsed={time.perf_counter() - group_t0:.3f}s'
            )
            # indices_flat: [B*P, N_max]
            N_max = indices_flat.shape[-1]

            # Reshape back to [B, P, N_max]
            indices_grouped = indices_flat.reshape(len(items), P, N_max)

            # Trim padded indices per sample
            for local_idx in range(len(items)):
                orig_idx = orig_indices[local_idx]
                pp_nf = per_person_num_frames_list[orig_idx]
                T_real = real_lengths[local_idx]

                if single_person_flags[local_idx]:
                    # Single-person: trim uniformly
                    N_real = self._temporal_to_token_len(T_real, T_max, N_max)
                    idx = indices_grouped[local_idx, 0, :N_real]  # [N_real]
                    results[orig_idx] = idx
                elif pp_nf is not None and any(t != T_real for t in pp_nf):
                    # Multi-person with different per-person lengths:
                    # trim each person independently.
                    per_person_indices = []
                    for p_idx in range(P):
                        T_p = min(pp_nf[p_idx], T_real) if p_idx < len(pp_nf) else T_real
                        if T_p <= 0:
                            # Person has no real content in this window (all padding).
                            per_person_indices.append(
                                indices_grouped.new_empty(0, dtype=torch.long)
                            )
                        else:
                            N_p = self._temporal_to_token_len(T_p, T_max, N_max)
                            per_person_indices.append(indices_grouped[local_idx, p_idx, :N_p])
                    results[orig_idx] = per_person_indices
                else:
                    # Multi-person, all persons share the same length: trim uniformly
                    N_real = self._temporal_to_token_len(T_real, T_max, N_max)
                    idx = indices_grouped[local_idx, :, :N_real]  # [P, N_real]
                    results[orig_idx] = idx

            self._debug_phase_log(
                f'batch_encode group trim done P={P} N_max={N_max} '
                f'elapsed={time.perf_counter() - group_t0:.3f}s'
            )

        return results

    def _wan_temporal_out(self, T_in: int) -> int:
        """Compute the number of output temporal tokens for T_in input frames.

        The WAN causal encoder processes input in chunks:
        chunk 0 = 1 frame, chunk i>0 = 4 frames.  Number of chunks =
        ``1 + max(0, (T_in - 1) // 4)``.

        * If the encoder has **temporal downsampling**, each chunk produces
          exactly 1 output frame → ``T_out = num_chunks``.
        * If the encoder has **no temporal downsampling** (e.g. 1D VQ-VAE with
          ``temporal_downsample=(False, False, False)``), each chunk preserves
          its input length.  Chunk 0 produces 1 frame, each subsequent chunk
          produces 4 frames → ``T_out = 1 + num_chunks_after_first * 4``.
          Note: trailing frames that don't fill a complete 4-frame chunk are
          **not consumed** by the encoder.
        """
        num_chunks = 1 + max(0, (T_in - 1) // 4)
        if self._motion_has_temporal_ds:
            return num_chunks
        else:
            # Each chunk preserves temporal length:
            # chunk 0 → 1 frame, chunks 1..(num_chunks-1) → 4 frames each
            return 1 + (num_chunks - 1) * 4

    def _temporal_to_token_len(self, T_real: int, T_padded: int, N_padded: int) -> int:
        """Compute output token length for T_real input frames.

        The per-timestep token count (K joints for 2-D, num_quantizers for
        1-D RVQ, or 1 for 1-D single-codebook) is inferred from the padded
        output so we don't need to hardcode model-specific constants.
        """
        if T_real == T_padded:
            return N_padded
        T_out_padded = self._wan_temporal_out(T_padded)
        T_out_real = self._wan_temporal_out(T_real)
        assert N_padded % T_out_padded == 0, (
            f"N_padded={N_padded} not divisible by T_out_padded={T_out_padded}"
        )
        tokens_per_ts = N_padded // T_out_padded
        return T_out_real * tokens_per_ts

    def motion_indices_to_lm_ids(self, indices, modal: Modality) -> Tensor:
        """Convert VQ codebook indices to LLM token IDs via lookup table.

        Args:
            indices: One of:
                - ``Tensor [N]`` (single-person)
                - ``Tensor [P, N]`` (multi-person, uniform length)
                - ``List[Tensor]`` of P tensors each ``[N_p]`` (multi-person,
                  per-person variable length from pseudo-composition trimming)
            modal: The motion modality class.

        Returns:
            1-D ``LongTensor`` of LLM token IDs with modal bos/eos and separators.
        """
        # Handle list-of-tensors (per-person variable length)
        if isinstance(indices, list):
            dev = indices[0].device
            bos_t = self._modal_bos_tensor(modal.name, dev)
            eos_t = self._modal_eos_tensor(modal.name, dev)
            lut = self._buffer_on_device(self.motion_vq_to_lm, dev)
            sep_t = self._buffer_on_device(self._mp_separator_t, dev)
            parts: List[Tensor] = [bos_t]
            num_emitted = 0
            for p_indices in indices:
                if p_indices.numel() == 0:
                    continue
                if num_emitted > 0:
                    parts.append(sep_t)
                parts.append(lut[p_indices])
                num_emitted += 1
            parts.append(eos_t)
            return torch.cat(parts)

        dev = indices.device
        bos_t = self._modal_bos_tensor(modal.name, dev)
        eos_t = self._modal_eos_tensor(modal.name, dev)
        lut = self._buffer_on_device(self.motion_vq_to_lm, dev)

        if indices.ndim == 1 or (indices.ndim == 2 and indices.shape[0] == 1):
            lm_ids = lut[indices.view(-1)]
            return torch.cat([bos_t, lm_ids, eos_t])
        else:
            sep_t = self._buffer_on_device(self._mp_separator_t, dev)
            parts: List[Tensor] = [bos_t]
            for p_idx in range(indices.shape[0]):
                if p_idx > 0:
                    parts.append(sep_t)
                parts.append(lut[indices[p_idx]])
            parts.append(eos_t)
            return torch.cat(parts)

    def _modal_data_to_lm_ids(
        self, modal: Modality, modal_data, encoded_motions: Dict, key: Tuple,
        device: torch.device = None,
    ) -> Tensor:
        """Convert a single modal's data to LLM token IDs (fast path).

        For motion modals with pre-encoded indices, uses lookup table (stays on GPU).
        For text-like modals, tokenizes the short string.
        For audio, falls back to string path.

        Returns:
            1-D ``LongTensor`` of LLM token IDs with modal bos/eos.
        """
        if device is None:
            device = self._runtime_device()

        if is_modal(modal, Motion) and key in encoded_motions:
            return self.motion_indices_to_lm_ids(encoded_motions[key], modal)

        bos_t = self._modal_bos_tensor(modal.name, device)
        eos_t = self._modal_eos_tensor(modal.name, device)

        if is_modal(modal, Audio):
            # Audio still uses string path
            modal_string = self.modal2string(modal, modal_data)
            content_ids = self.text_tokenizer.encode(modal_string, add_special_tokens=False)
            return self._ids_to_runtime_device(content_ids, device)

        # Text-like modals: tokenize short strings
        if is_modal(modal, Duration):
            text = f"{float(modal_data):.1f}"
        elif is_modal(modal, NumPerson):
            text = f"{int(modal_data)}"
        elif is_modal(modal, Text):
            text = format_text_modal_data(modal, modal_data)
        else:
            # Fallback for any unknown modal
            modal_string = self.modal2string(modal, modal_data)
            content_ids = self.text_tokenizer.encode(modal_string, add_special_tokens=False)
            return self._ids_to_runtime_device(content_ids, device)

        content_ids = self.text_tokenizer.encode(text, add_special_tokens=False)
        content_t = self._ids_to_runtime_device(content_ids, device)
        return torch.cat([bos_t, content_t, eos_t])

    def build_mm_tokenizer(self, mm_tokenizer_cfg: Dict) -> nn.Module:
        """
        :param mm_tokenizer_cfg: tokenizer config
        :return: tokenizer module.
        """
        type = mm_tokenizer_cfg["type"]
        if os.path.isfile(type):
            # allow using config file path as type, simplify config writing.
            init_cfg = mm_tokenizer_cfg.pop("init_cfg", None)
            mm_tokenizer_cfg = Config.fromfile(type)["model"]
            if init_cfg is not None:
                mm_tokenizer_cfg["init_cfg"] = init_cfg

        mm_tokenizer: nn.Module = MODELS.build(mm_tokenizer_cfg).eval()
        if mm_tokenizer_cfg.get("init_cfg", None) is not None:
            mm_tokenizer.init_weights()
        return mm_tokenizer

    def modal2string(self, modal: Modality, modal_data) -> str:
        """
        Convert modal data to string.

        :param modal: modality
        :param modal_data: modal data.
        :return: modal string
        """
        if is_modal(modal, Motion):
            assert isinstance(
                modal_data, torch.Tensor
            ), f"Invalid modal {modal.name} modal data: {modal_data}"
            modal_string = self.motion2string(modal_data)
        elif is_modal(modal, Audio):
            assert isinstance(
                modal_data, torch.Tensor
            ), f"Invalid modal {modal.name} modal data: {modal_data}"
            modal_string = self.audio2string(modal_data.unsqueeze(0))
        elif is_modal(modal, Duration):
            modal_string = f"{float(modal_data):.1f}"
        elif is_modal(modal, NumPerson):
            modal_string = f"{int(modal_data)}"
        else:
            # for text data, no need to additional processing
            assert is_modal(modal, Text), f"Invalid modal: {modal}, {modal_data}"
            assert isinstance(modal_data, (str, list, tuple)), (
                f"Invalid modal {modal.name} modal data: {modal_data}"
            )
            modal_string = format_text_modal_data(modal, modal_data)

        modal_string = modal.bos + modal_string + modal.eos
        return modal_string

    @torch.no_grad()
    def audio2string(self, audio: Tensor) -> str:
        if self.audio_tokenizer is None:
            raise RuntimeError(
                "audio_tokenizer is not loaded. Set audio_tokenizer in config "
                "to use audio tasks."
            )
        audio = audio.to(self._runtime_device(), non_blocking=True)
        audio_ids = self.audio_tokenizer.encode(audio)[1].squeeze(0)
        audio_strings = Audio.index_to_string(audio_ids)
        return audio_strings

    @torch.no_grad()
    def motion2string(self, motion: Tensor) -> str:
        """Quantize the motion vector into indexes with tokenizer, and save the idx to data_samples.
        Some idx will be used in Completion tasks.
        :param motion: A list of motion samples, each sample is a tensor of shape [t, c] or [p, t, c]. We don't do any padding here to avoid complex logic.
        :param data_samples: data sample for a single sample(not a batch)
        :return: indexes, motion strings, data_samples
        """
        # encode motion to motion_ids
        # [t, c] or [p, t, c] -> [p, n] or [1, n] -> [n]
        # TODO: need to update if VersatileMotion supports mixed training with different num_persons within a batch
        assert motion.ndim in [2, 3], f"Invalid motion shape: {motion.shape}"
        num_person = 1 if motion.ndim == 2 else int(motion.shape[0])
        smpl_pose_processor = self._smpl_processor_for_num_person(num_person)
        motion = motion.to(self._runtime_device(), non_blocking=True)
        if motion.ndim == 2:
            motion = motion.unsqueeze(0)  # [t c] to [b t c]
        transl, pose = motion[..., :6], motion[..., 6:]
        transl = smpl_pose_processor.inv_convert_transl(transl)
        if self.motion_tokenizer.config.use_static:
            joints = smpl_pose_processor.fk(
                transl,
                pose,
            )
            static_joints = smpl_pose_processor.get_static_joint_mask(
                joints[..., [7, 10, 8, 11, 20, 21], :],
                vel_thr=0.15,
                repeat_last=True,
            )

        motion = smpl_pose_processor.normalize(motion)
        if self.motion_tokenizer.config.use_static:
            motion = torch.cat([motion, static_joints], dim=-1)

        if isinstance(self.motion_tokenizer, VQVAEWanMotion1D):
            # 1D VQ-VAE: input [B, T, C]
            motion_ids = self.motion_tokenizer.encode(motion).indices
            if motion_ids.ndim == 3:
                # RVQ: [B, N, num_q] → offset each layer and interleave → [B, N*num_q]
                B, N, Q = motion_ids.shape
                cb = self.motion_tokenizer.codebook_size
                offsets = torch.arange(Q, device=motion_ids.device) * cb
                motion_ids = (motion_ids + offsets).reshape(B, N * Q)
            motion_ids = motion_ids.squeeze(0)
        else:
            # 2D VQ-VAE: input [B, T, K, C]
            motion = rearrange(motion, "b t (j d) -> b t j d", d=6)
            motion_ids = self.motion_tokenizer.encode(motion, flatten=True).indices.squeeze(0)
        motion_string = Motion.index_to_string(motion_ids)

        return motion_string

    @torch.no_grad()
    def string2audio(self, audio_string: List[str], modal: Modality) -> Tensor:
        """Convert audio string to audio tensor
        :param audio_string: audio string
        :return: audio tensor
        """
        if self.audio_tokenizer is None:
            raise RuntimeError(
                "audio_tokenizer is not loaded. Set audio_tokenizer in config "
                "to use audio tasks."
            )
        audio_ids = [modal.string_to_index(t) for t in audio_string]
        audio = [
            (
                self.audio_tokenizer.decode(
                    ids.unsqueeze(0).to(self._runtime_device()), is_idx=True
                ).squeeze(0)
                if ids is not None
                else None
            )
            for ids in audio_ids
        ]
        return audio

    def fill_chat_template(
        self, task_template: str, condition_parts: List[str], output_parts: List[str]
    ) -> str:
        """
        :param task_template: A task template(string)
        :param condition_parts: A list of condition parts, each part is a string
        :param output_parts: A list of output parts, each part is a string
        """

        # 随机打乱并拼接 condition / output
        if self.shuffle_condition_parts:
            random.shuffle(condition_parts)
        condition_str = "".join(condition_parts)

        if self.shuffle_output_parts:
            random.shuffle(output_parts)
        output_str = "".join(output_parts)

        # ---- 把你原来的结构塞到 chat messages 里 ----
        # user 这段：task_bos + 模板 + task_eos + cond_bos + 条件 + cond_eos
        user_content = (
            self.task_bos
            + task_template
            + self.task_eos
            + self.cond_bos
            + condition_str
            + self.cond_eos
        )

        # assistant 这段：output_bos + 输出内容
        assistant_content = self.output_bos + output_str

        messages = [
            # {
            #     "role": "system",
            #     "content": self.system_message(),
            # },
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": assistant_content,
            },
        ]

        # 关键：用 tokenizer 的 chat_template 生成完整文本
        # tokenize=False → 返回字符串（里面已经包含所有 special token）
        # add_generation_prompt=False → 因为你已经把 assistant 回复也放进来了
        text = self.text_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, truncation=False
        )

        return text

    def encode_text(
        self,
        texts: List[str],
        return_labels: bool = False,
        instruction_stage: bool = False,
    ) -> Tensor:
        # 如果你是自己手写 ChatML，可以在这里视情况去掉末尾多余换行
        # texts = [t.rstrip("\n") for t in texts]

        lm_input = self.text_tokenizer(
            texts,
            add_special_tokens=False,  # 你已自己插入 special tokens
            padding=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_attention_mask=True,
            return_tensors="pt",
            truncation=False,
        ).to(self._runtime_device())

        if return_labels:
            input_ids = lm_input["input_ids"]
            labels = input_ids.clone()

            # 1) pad 位置不计算 loss
            labels[labels == self.text_tokenizer.pad_token_id] = -100

            if instruction_stage:
                # 2) 只在 instruction 阶段，用 out_bos 决定监督起点
                bos_id = self.output_bos_id

                # (B, L) -> True/False，标记每个位置是否是 out_bos
                bos_mask = input_ids == bos_id

                if bos_mask.any():
                    B, L = input_ids.shape
                    positions = (
                        torch.arange(L, device=input_ids.device)
                        .unsqueeze(0)
                        .expand(B, L)
                    )

                    # 对没有 out_bos 的样本，我们保持 last_pos = -1
                    # torch.where: 有 bos 的地方用真实位置，否则为 -1
                    bos_pos = torch.where(
                        bos_mask, positions, torch.full_like(positions, -1)
                    )
                    # 每条样本里最后一个 out_bos 的位置
                    last_bos_pos, _ = bos_pos.max(dim=1)  # (B,)
                    has_bos = last_bos_pos >= 0  # (B,)

                    # 构造 mask: 对于有 bos 的样本，把 <= last_bos_pos 的位置设为 -100
                    mask_before_bos = positions <= last_bos_pos.unsqueeze(1)  # (B, L)
                    mask_before_bos &= has_bos.unsqueeze(1)  # 只有有 bos 的行才生效

                    labels[mask_before_bos] = -100

                    # 如果你不想让 out_bos 自己算 loss，可以再把等于 bos_id 的位置也置成 -100：
                    # labels[input_ids == bos_id] = -100
            lm_input["labels"] = labels

        return lm_input

    def sample_optional_input_modals(
        self, optional_input_modals: List[Modality], mode: str = None
    ):
        if mode is None:
            mode = self.optional_input_modal_mode

        if mode == "none":
            return []
        elif mode == "all":
            return optional_input_modals
        elif mode == "duration":
            if Duration in optional_input_modals:
                return [Duration]
            else:
                return []
        elif mode == "caption":
            if Caption in optional_input_modals:
                return [Caption]
            else:
                return []
        elif mode == "random":
            num_optional_input = random.randint(0, len(optional_input_modals))
            return random.sample(optional_input_modals, num_optional_input)
        else:
            raise ValueError(f"Invalid optional input modal mode: {mode}")

    def process_train(self, inputs: Dict):
        debug_t0 = time.perf_counter()
        tasks: List[BaseTask] = inputs["task"]
        batch_size = len(tasks)
        task_abbrs = [getattr(task, 'abbr', str(task)) for task in tasks]
        self._debug_phase_log(
            f'process_train start batch_size={batch_size} tasks={task_abbrs}'
        )

        # ── Phase 1: Collect all motion tensors and batch-encode ────────
        # Key = (sample_idx, modal_name) so we can look them up later.
        motion_keys: List[Tuple[int, str]] = []
        motion_tensors: List[Tensor] = []
        # Per-person frame counts for each motion tensor (None if uniform).
        motion_pp_nf_list: List[Optional[List[int]]] = []

        # We need to know per-sample: template, selected optional modals, output modals.
        per_sample_info: List[dict] = []

        for sample_idx, task in enumerate(tasks):
            template: str = (
                task.templates[0]
                if self.task_template_mode == "first"
                else random.choice(task.templates)
            )
            optional_input_modals: List[Modality] = self.sample_optional_input_modals(
                task.optional_input_modality
            )

            # Force Caption when condition motion has very few frames
            # (pred/inbetween with single-frame conditions need text guidance)
            if task.abbr in ("pred", "inbetween") and Caption not in optional_input_modals:
                force_caption = False
                past = obtain_data(PastMotion, inputs, sample_idx, allow_none=True)
                if past is not None and past.shape[-2] <= 1:
                    force_caption = True
                if task.abbr == "inbetween":
                    future = obtain_data(FutureMotion, inputs, sample_idx, allow_none=True)
                    if future is not None and future.shape[-2] <= 1:
                        force_caption = True
                if force_caption and Caption in task.optional_input_modality:
                    optional_input_modals.append(Caption)

            output_modals: List[Modality] = task.output_modality

            # Collect input condition modals (required + optional)
            condition_modals: List[Modality] = list(task.input_modality)
            # Filter optional modals that actually have data
            valid_optional: List[Modality] = []
            for modal in optional_input_modals:
                data = obtain_data(modal, inputs, sample_idx, allow_none=True)
                if data is not None:
                    valid_optional.append(modal)
            condition_modals.extend(valid_optional)

            per_sample_info.append(dict(
                template=template,
                condition_modals=condition_modals,
                output_modals=output_modals,
            ))

            # Collect all motion tensors (from conditions + outputs)
            for modal in condition_modals + output_modals:
                if is_modal(modal, Motion):
                    data = obtain_data(modal, inputs, sample_idx, allow_none=False)
                    if isinstance(data, torch.Tensor) and data.shape[-2] > 0:
                        key = (sample_idx, modal.name)
                        motion_keys.append(key)
                        motion_tensors.append(data)
                        # Look up per-person frame counts for this motion modal.
                        # e.g. modal "motion" -> "per_person_num_frames",
                        #      modal "past_motion" -> "past_per_person_num_frames"
                        data_key = modal.data_keys[0]  # e.g. "motion", "past_motion"
                        if data_key == "motion":
                            pp_key = "per_person_num_frames"
                        else:
                            # "past_motion" -> "past_per_person_num_frames"
                            prefix = data_key.rsplit("_motion", 1)[0]
                            pp_key = f"{prefix}_per_person_num_frames"
                        pp_nf = None
                        if pp_key in inputs:
                            pp_data = inputs[pp_key]
                            if isinstance(pp_data, torch.Tensor) and pp_data.ndim >= 2:
                                pp_nf = pp_data[sample_idx].tolist()
                            elif isinstance(pp_data, (list, tuple)) and sample_idx < len(pp_data):
                                val = pp_data[sample_idx]
                                if val is not None:
                                    if isinstance(val, torch.Tensor):
                                        pp_nf = val.tolist()
                                    elif isinstance(val, list):
                                        pp_nf = val
                                    else:
                                        pp_nf = list(val)
                        motion_pp_nf_list.append(pp_nf)

        # Batch-encode all motions (padded + batched for efficiency)
        encoded_motions: Dict[Tuple[int, str], Tensor] = {}
        motion_shapes = [tuple(t.shape) for t in motion_tensors]
        path_keys = ['motion_path', 'smplx_path', 'id', 'key']
        meta_bits = []
        for meta_key in path_keys:
            if meta_key in inputs:
                value = inputs[meta_key]
                text = str(value)
                if len(text) > 300:
                    text = text[:300] + '...'
                meta_bits.append(f'{meta_key}={text}')
        self._debug_phase_log(
            f'process_train collected motions={len(motion_tensors)} '
            f'shapes={motion_shapes[:8]} elapsed={time.perf_counter() - debug_t0:.3f}s '
            f'meta={"; ".join(meta_bits)}'
        )
        if len(motion_tensors) > 0:
            encoded_list = self.batch_encode_motion(motion_tensors, motion_pp_nf_list)
            self._debug_cuda_sync()
            encoded_shapes = [
                [tuple(x.shape) for x in item]
                if isinstance(item, list) else tuple(item.shape)
                for item in encoded_list
            ]
            self._debug_phase_log(
                f'process_train batch_encode done encoded={len(encoded_list)} '
                f'shapes={encoded_shapes[:8]} elapsed={time.perf_counter() - debug_t0:.3f}s'
            )
            for key, indices in zip(motion_keys, encoded_list):
                encoded_motions[key] = indices
            del encoded_list

        # Free raw motion tensors — only the small VQ index tensors are kept
        del motion_tensors, motion_keys, motion_pp_nf_list

        # ── Phase 2: Build token ID sequences on the runtime device ─────────
        device = self._runtime_device()
        batch_seq_tensors: List[Tensor] = []
        output_bos_positions: List[int] = []

        for sample_idx, task in enumerate(tasks):
            info = per_sample_info[sample_idx]
            template = info["template"]
            condition_modals = info["condition_modals"]
            output_modals = info["output_modals"]

            # Build condition token IDs
            condition_id_parts: List[Tensor] = []
            for modal in condition_modals:
                data = obtain_data(modal, inputs, sample_idx, allow_none=True)
                if data is None:
                    continue
                key = (sample_idx, modal.name)
                ids = self._modal_data_to_lm_ids(modal, data, encoded_motions, key, device=device)
                condition_id_parts.append(ids)

            # Build output token IDs
            output_id_parts: List[Tensor] = []
            for modal in output_modals:
                data = obtain_data(modal, inputs, sample_idx, allow_none=False)
                key = (sample_idx, modal.name)
                ids = self._modal_data_to_lm_ids(modal, data, encoded_motions, key, device=device)
                output_id_parts.append(ids)

            # Conditions may be order-augmented, while outputs keep a stable
            # task-defined grammar so modal transitions have one target.
            if self.shuffle_condition_parts:
                random.shuffle(condition_id_parts)
            if self.shuffle_output_parts:
                random.shuffle(output_id_parts)

            # Tokenize the task template text
            template_ids = self.text_tokenizer.encode(template, add_special_tokens=False)
            template_t = self._ids_to_runtime_device(template_ids, device)

            # Assemble the full sequence following the chat template structure:
            # <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
            # {task_bos}{template}{task_eos}{cond_bos}{conditions}{cond_eos}
            # <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
            # {output_bos}{outputs}<|eot_id|>
            parts: List[Tensor] = []

            # User turn
            parts.append(self._buffer_on_device(self._chat_user_prefix_t, device))
            parts.append(self._buffer_on_device(self._task_bos_t, device))
            parts.append(template_t)
            parts.append(self._buffer_on_device(self._task_eos_t, device))
            parts.append(self._buffer_on_device(self._cond_bos_t, device))
            parts.extend(condition_id_parts)
            parts.append(self._buffer_on_device(self._cond_eos_t, device))
            parts.append(self._buffer_on_device(self._chat_user_suffix_t, device))

            # Assistant turn
            parts.append(self._buffer_on_device(self._chat_asst_prefix_t, device))
            output_bos_pos = sum(part.numel() for part in parts)
            parts.append(self._buffer_on_device(self._output_bos_t, device))
            parts.extend(output_id_parts)
            parts.append(self._buffer_on_device(self._chat_asst_suffix_t, device))

            seq_t = torch.cat(parts)
            assert seq_t.numel() > 0
            batch_seq_tensors.append(seq_t)
            output_bos_positions.append(output_bos_pos)
            self._debug_phase_log(
                f'process_train sample_seq sample={sample_idx} len={seq_t.numel()} '
                f'elapsed={time.perf_counter() - debug_t0:.3f}s'
            )

        # Free VQ index tensors — Phase 2 has already converted them to LM token IDs
        del encoded_motions

        # ── Phase 2.5: Truncate over-long sequences ─────────────────────
        if self.max_seq_len > 0:
            for i in range(len(batch_seq_tensors)):
                if batch_seq_tensors[i].numel() > self.max_seq_len:
                    seq_t = batch_seq_tensors[i]
                    if self.instruction_stage:
                        bos_pos = output_bos_positions[i]
                        output_len = seq_t.numel() - bos_pos
                        if output_len >= self.max_seq_len:
                            batch_seq_tensors[i] = seq_t[
                                bos_pos : bos_pos + self.max_seq_len
                            ]
                            output_bos_positions[i] = 0
                        else:
                            prefix_budget = self.max_seq_len - output_len
                            prefix_start = max(0, bos_pos - prefix_budget)
                            batch_seq_tensors[i] = torch.cat(
                                [seq_t[prefix_start:bos_pos], seq_t[bos_pos:]]
                            )
                            output_bos_positions[i] = bos_pos - prefix_start
                        continue
                    batch_seq_tensors[i] = seq_t[: self.max_seq_len]
                    output_bos_positions[i] = min(output_bos_positions[i], self.max_seq_len - 1)
        else:
            self._debug_phase_log('process_train max_seq_len disabled')

        # ── Phase 3: Right-pad, attention mask, labels ─────────────────
        seq_lengths = [t.numel() for t in batch_seq_tensors]
        max_len = max(seq_lengths)
        needs_padding = any(seq_len != max_len for seq_len in seq_lengths)
        pad_id = self.text_tokenizer.pad_token_id
        input_ids, attention_mask, labels = pad_training_token_sequences(
            batch_seq_tensors,
            pad_id,
            self.instruction_stage,
            output_bos_positions,
        )

        lm_input_dict = {
            "input_ids": input_ids,
            "labels": labels,
        }
        if attention_mask is not None:
            lm_input_dict["attention_mask"] = attention_mask
        lm_input = BatchEncoding(lm_input_dict)
        self._debug_cuda_sync()
        self._debug_phase_log(
            f'process_train done input_shape={tuple(input_ids.shape)} '
            f'max_len={max_len} needs_padding={needs_padding} '
            f'elapsed={time.perf_counter() - debug_t0:.3f}s'
        )
        return lm_input

    def locate_modal(self, batch_text: List[str]) -> Dict:
        """
            1, Firstly, fetch special tokens from the output of causal lm
            2, The predicted modal of each sample may differ in each batch,
            once a sample A has modal X, other samples should make dummy index
            to keep synchronization with sample A. We use [0] as the dummy index
        :param batch_text: causal lm predicted text
        :return: Modality -> corresponding sub string in the LLM output text.
        """
        batch_modal_dict = defaultdict(list)
        for text in batch_text:
            for modal in LOCATABLE_MODALS:
                match_text = modal.locate_modality(text)
                if len(match_text) and len(match_text[0]):
                    batch_modal_dict[modal].append(match_text[0])
                else:
                    batch_modal_dict[modal].append(None)
        return batch_modal_dict

    def idx2text(self, ids: Tensor) -> List[str]:
        return self.text_tokenizer.batch_decode(ids, skip_special_tokens=False)
