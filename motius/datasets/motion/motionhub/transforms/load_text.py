import os
import random
from typing import Dict, List, Optional, Tuple, Union
from mmcv.transforms import BaseTransform

import torch

from motius.datasets.motion.motionhub.common import hm3d_pattern, read_json, read_txt
from pathlib import Path

from motius.registry import TRANSFORMS


def _motionhub_qwen3_root() -> str:
    env_root = os.environ.get('MOTIUS_MOTIONHUB_QWEN3_ROOT')
    if env_root:
        return env_root
    # load_text.py -> transforms -> motionhub -> motion -> datasets -> motius -> repo
    return str(Path(__file__).resolve().parents[5] / 'data' / 'motionhub_qwen3')


# ---------------------------------------------------------------------------
# Mapping: caption dir name → pre-extracted qwen3 embedding dir name.
# When a caption JSON file lives under dir X, the corresponding .pt embedding
# file (if it exists) lives under the sibling dir CAPTION_TO_QWEN3_DIR[X].
# The .pt file has the same relative path as the .json but with .pt suffix.
# ---------------------------------------------------------------------------
CAPTION_TO_QWEN3_DIR = {
    # Academic / AcademicRetarget / Game / Taobao (and their mirror variants)
    'human_checked_augmented_caption': 'qwen3_human_checked_short',
    'human_checked_augmented_caption_deprecated_mirror_251215': 'qwen3_human_checked_short',
    'human_checked_augmented_caption_mirror': 'qwen3_human_checked_short',
    'human_checked_caption': 'qwen3_human_checked_short',
    'human_checked_caption_deprecated_mirror_251215': 'qwen3_human_checked_short',
    'human_checked_caption_mirror': 'qwen3_human_checked_short',
    'improved_simple_augmented_caption': 'qwen3_improved_simple_short',
    'improved_simple_augmented_caption_deprecated_mirror_251215': 'qwen3_improved_simple_short',
    'improved_simple_caption': 'qwen3_improved_simple_short',
    'improved_simple_caption_deprecated_mirror_251215': 'qwen3_improved_simple_short',
    # PerMo / general augmented captions (Qwen3-8B CausalLM embeddings)
    'augmented_caption': 'qwen3_augmented',
    'augmented_caption_deprecated_250905': 'qwen3_augmented',
    'augmented_caption_deprecated_250926': 'qwen3_augmented',
    # PerMo editing instructions (Neutral→Emotion style transfer)
    'editing_caption': 'qwen3_editing',
    # MotionFix instructions keep the same nested split/bucket layout under
    # pairs/ and qwen3_editing/.
    'pairs': 'qwen3_editing',
    # MotionHub high-quality captions (macro/meso/micro hierarchy).
    'hierarchical_caption': 'qwen3_hierarchical',
}

DEFAULT_AUGMENT_SOURCE_DIRS = (
    'qwen3_human_checked_short',
    'qwen3_improved_simple_short',
)


def _caption_path_to_embedding_path(caption_path: str) -> Optional[str]:
    """Given an absolute caption .json path, return the corresponding .pt
    pre-extracted embedding path, or None if no mapping is known.

    The .pt file has identical structure to the embedding dict returned by
    HYTextModel.encode():
        data['result'][i] = {
            'caption': str,
            'text_embedding': {
                'text_vec_raw':        Tensor[1, 1, 768],
                'text_ctxt_raw':       Tensor[1, seq, 4096],
                'text_ctxt_raw_length': Tensor[1],
            },
            ...
        }
    """
    # Normalize the path first (resolve ../ segments)
    caption_path = os.path.normpath(caption_path)
    # Walk up the path to find the first path component that matches a known
    # caption dir name.
    parts = caption_path.replace('\\', '/').split('/')
    for i, part in enumerate(parts):
        if part in CAPTION_TO_QWEN3_DIR:
            qwen3_dir = CAPTION_TO_QWEN3_DIR[part]
            if part == 'hierarchical_caption':
                # MotionHub features are stored outside the source MotionHub tree:
                # data/motionhub/<subset>/hierarchical_caption/x.json
                #   -> data/motionhub_qwen3/<subset>/qwen3_hierarchical/x.pt
                try:
                    motionhub_idx = parts.index('motionhub')
                    rel_parts = parts[motionhub_idx + 1:]
                except ValueError:
                    rel_parts = parts[i - 1:] if i > 0 else parts[i:]
                for j, rel_part in enumerate(rel_parts):
                    if rel_part == part:
                        rel_parts[j] = qwen3_dir
                        break
                pt_path = os.path.join(_motionhub_qwen3_root(), *rel_parts)
            else:
                new_parts = parts[:i] + [qwen3_dir] + parts[i + 1:]
                pt_path = '/'.join(new_parts)
            # Replace .json → .pt
            if pt_path.endswith('.json'):
                pt_path = pt_path[:-5] + '.pt'
            return pt_path
    return None


def _replace_qwen3_dir(
    pt_path: str,
    qwen3_dir: str,
    allowed_source_dirs: Optional[Tuple[str, ...]] = None,
) -> Optional[str]:
    """Replace the qwen3 feature directory component in an embedding path."""
    parts = os.path.normpath(pt_path).replace('\\', '/').split('/')
    for i, part in enumerate(parts):
        if part.startswith('qwen3_'):
            if allowed_source_dirs is not None and part not in allowed_source_dirs:
                return None
            if part == qwen3_dir:
                return None
            parts[i] = qwen3_dir
            return '/'.join(parts)
    return None


@TRANSFORMS.register_module(force=True)
class LoadPreExtractedTextEmbedding(BaseTransform):
    """Load pre-extracted Qwen3+CLIP text embeddings from .pt files.

    For each sample, the caption JSON path (``results['caption_path']``) is
    mapped to a sibling .pt file that contains pre-extracted embeddings.
    If the .pt file exists, the embeddings are loaded directly (bypassing
    online Qwen3-8B inference during training).  If no .pt file is found,
    the transform falls back gracefully: it leaves ``results`` unchanged so
    that the downstream trainer can fall back to online encoding or null
    embeddings.

    The .pt file format expected::

        data['result'][i] = {
            'caption': str,
            'text_embedding': {
                'text_vec_raw':         Tensor[1, 1, 768],   # CLIP-L
                'text_ctxt_raw':        Tensor[1, seq, 4096], # Qwen3
                'text_ctxt_raw_length': Tensor[1],
            },
            ...
        }

    Output keys added to results (when successful):
        ``text_vec_raw``, ``text_ctxt_raw``, ``text_ctxt_raw_length``
    The ``caption`` string is also set (from the chosen embedding item) so
    that existing CFG dropout logic continues to work.

    Args:
        key (str): Key prefix for caption path in results dict.
            Default ``'caption'`` → reads ``results['caption_path']``.
        allow_none (bool): If True, silently skip when caption_path is None.
        fallback_to_caption (bool): If True (default), keep ``caption`` text
            in results even when embedding is found, so text-only fallback
            pipelines still work.
        text_emb_augment_dir (str, optional): Optional sibling qwen3 feature
            directory.  When provided and the augmented file exists, raw and
            augmented features are sampled with the same 50/50 policy as the
            official HY-Motion T2M dataset.
        text_emb_long_dir (str, optional): Optional sibling long-caption
            feature directory.  Long features are appended to the augmented
            candidate pool, matching the official dataset behavior.
        refetch_on_missing (bool): If True, return None when no extracted
            feature is available so the dataset can resample the record.
        augment_source_dirs (tuple[str], optional): Raw qwen3 dirs that are
            allowed to look up augmented/long siblings.  Defaults to the two
            official T2M raw dirs so edit-instruction features from
            ``qwen3_editing`` are never replaced by generic T2M captions.
        raw_text_prob (float): Probability of sampling the raw feature pool
            when both raw and augmented/long pools are non-empty.
    """

    def __init__(
        self,
        key: str = 'caption',
        allow_none: bool = True,
        fallback_to_caption: bool = True,
        vtxt_dim: int = 768,
        ctxt_dim: int = 4096,
        text_emb_augment_dir: Optional[str] = None,
        text_emb_long_dir: Optional[str] = None,
        refetch_on_missing: bool = False,
        augment_source_dirs: Optional[Tuple[str, ...]] = DEFAULT_AUGMENT_SOURCE_DIRS,
        raw_text_prob: float = 0.5,
    ):
        self.key = key
        self.allow_none = allow_none
        self.fallback_to_caption = fallback_to_caption
        self.vtxt_dim = vtxt_dim
        self.ctxt_dim = ctxt_dim
        self.text_emb_augment_dir = text_emb_augment_dir
        self.text_emb_long_dir = text_emb_long_dir
        self.refetch_on_missing = bool(refetch_on_missing)
        self.augment_source_dirs = (
            tuple(augment_source_dirs) if augment_source_dirs is not None else None)
        self.raw_text_prob = float(raw_text_prob)

        if not 0.0 <= self.raw_text_prob <= 1.0:
            raise ValueError(f'raw_text_prob must be in [0, 1], got {raw_text_prob}')

    def _fill_null_embedding(self, results: Dict) -> Dict:
        """Fill null (zero) embedding tensors so that every sample in a batch
        has a consistent tensor type for collation.  The trainer's CFG dropout
        (mask_text_cond) will replace these with the *learned* null embeddings
        when cond_mask_prob triggers, but having zeros here prevents mixed
        Tensor/None collation errors.

        Note: we mark null samples with text_ctxt_raw_length=0 so the trainer
        can build a correct attention mask (all False → padding).
        """
        results['text_vec_raw'] = torch.zeros(1, self.vtxt_dim)
        results['text_ctxt_raw'] = torch.zeros(1, self.ctxt_dim)
        results['text_ctxt_raw_length'] = torch.tensor(0)
        results['_text_is_null'] = True
        return results

    def _handle_missing_embedding(self, results: Dict) -> Optional[Dict]:
        if self.refetch_on_missing:
            return None
        return self._fill_null_embedding(results)

    def _load_result_list(self, pt_path: Optional[str]) -> List[Dict]:
        if pt_path is None or not os.path.exists(pt_path):
            return []
        try:
            data = torch.load(pt_path, map_location='cpu', weights_only=True)
        except TypeError:
            try:
                data = torch.load(pt_path, map_location='cpu')
            except Exception:
                return []
        except Exception:
            return []

        result_list = data.get('result', []) if isinstance(data, dict) else []
        return result_list if isinstance(result_list, list) else []

    def _choose_feature_pool(
        self,
        raw_features: List[Dict],
        augment_features: List[Dict],
        long_features: List[Dict],
    ) -> Tuple[List[Dict], str, int]:
        aug_long_features = augment_features + long_features
        if not raw_features and not aug_long_features:
            return [], 'null', -1

        if not raw_features:
            choose_raw = False
        elif not aug_long_features:
            choose_raw = True
        else:
            choose_raw = random.random() < self.raw_text_prob

        if choose_raw:
            return raw_features, 'raw', -1

        if augment_features and long_features:
            source_type = 'augment_or_long'
        elif augment_features:
            source_type = 'augment'
        else:
            source_type = 'long'
        return aug_long_features, source_type, len(augment_features)

    def transform(self, results: Dict) -> Optional[Dict]:
        caption_path = results.get(f'{self.key}_path')
        if caption_path is None:
            if self.allow_none:
                return self._handle_missing_embedding(results)
            raise ValueError(
                f"LoadPreExtractedTextEmbedding: '{self.key}_path' not found in results"
            )

        # Derive .pt path from caption JSON path
        pt_path = _caption_path_to_embedding_path(caption_path)
        raw_features = self._load_result_list(pt_path)

        augment_features: List[Dict] = []
        if self.text_emb_augment_dir and pt_path is not None:
            augment_pt_path = _replace_qwen3_dir(
                pt_path, self.text_emb_augment_dir, self.augment_source_dirs)
            augment_features = self._load_result_list(augment_pt_path)

        long_features: List[Dict] = []
        if self.text_emb_long_dir and pt_path is not None:
            long_pt_path = _replace_qwen3_dir(
                pt_path, self.text_emb_long_dir, self.augment_source_dirs)
            long_features = self._load_result_list(long_pt_path)

        result_list, source_type, augment_count = self._choose_feature_pool(
            raw_features, augment_features, long_features)
        if not result_list:
            return self._handle_missing_embedding(results)

        # Randomly select one caption variant (data augmentation)
        idx = random.randint(0, len(result_list) - 1)
        item = result_list[idx]
        emb = item.get('text_embedding')
        if emb is None:
            return self._handle_missing_embedding(results)

        # Unpack: remove the leading batch dim added during extraction
        # Each tensor was saved as [1, ...] from a batch-size-1 encode call.
        text_vec_raw = emb['text_vec_raw'].squeeze(0)          # [1, 768]
        text_ctxt_raw = emb['text_ctxt_raw'].squeeze(0)        # [seq, 4096]
        text_ctxt_raw_length = emb['text_ctxt_raw_length'].squeeze(0)  # scalar

        results['text_vec_raw'] = text_vec_raw
        results['text_ctxt_raw'] = text_ctxt_raw
        results['text_ctxt_raw_length'] = text_ctxt_raw_length
        results['_text_is_null'] = False
        results['text_idx'] = idx

        if source_type == 'augment_or_long':
            results['text_source_type'] = 'augment' if idx < augment_count else 'long'
        else:
            results['text_source_type'] = source_type

        start_time = item.get('start_time', 0.0)
        end_time = item.get('end_time', 0.0)
        try:
            start_time = float(start_time)
        except (TypeError, ValueError):
            start_time = 0.0
        try:
            end_time = float(end_time)
        except (TypeError, ValueError):
            end_time = 0.0
        if start_time != start_time:
            start_time = 0.0
        if end_time != end_time:
            end_time = 0.0
        results['caption_start_time'] = start_time
        results['caption_end_time'] = end_time

        # Also store caption string (for logging / CFG dropout compatibility)
        if self.fallback_to_caption:
            results['caption'] = item.get(
                'caption', item.get('short caption', results.get('caption', '')))

        return results


@TRANSFORMS.register_module(force=True)
class LoadPreExtractedT5Feature(BaseTransform):
    """Load pre-extracted T5 (UMT5) text embeddings from .pt files.

    Replaces online T5 encoding during PRISM training. At each call:
    1. Map caption_path → .pt feature path (under feature_dir)
    2. Load .pt, randomly select one variant's embedding
    3. Pad to max_seq_length with zeros (matches encode_prompt_with_mask behavior)
    4. Build attention mask (1s for valid tokens, 0s for padding)

    Prompt dropout is handled by the trainer (replaces embedding with
    pre-extracted null embedding at prompt_drop_rate probability).

    The .pt file format expected::

        {
            'captions': ['text1', 'text2', ...],
            'embeddings': [Tensor[L1, 4096], Tensor[L2, 4096], ...],  # bf16, unpadded
            'seq_lens': [int, int, ...],
        }

    Output keys added to results:
        ``t5_text_embeds``: Tensor[max_seq_length, 4096] bf16 (padded)
        ``t5_text_mask``:   Tensor[max_seq_length] int64 (1=valid, 0=pad)
        ``caption``:        str (the selected caption text, for logging)

    Args:
        feature_dir (str): Root directory of pre-extracted T5 features.
        data_dir (str): The dataset's data_dir (for path remapping).
        max_seq_length (int): Pad/truncate embeddings to this length.
        allow_none (bool): If True, return None when caption_path is missing
            (triggers dataset refetch). If False, raise error.
        hidden_dim (int): T5 hidden dimension (default 4096).
    """

    def __init__(
        self,
        feature_dir: str = 'data/t5_feature',
        data_dir: str = 'data/motionhub',
        max_seq_length: int = 256,
        allow_none: bool = True,
        hidden_dim: int = 4096,
        select_idx: Optional[int] = None,
        raw_text_prob: Optional[float] = None,
    ):
        self.feature_dir = feature_dir
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.allow_none = allow_none
        self.hidden_dim = hidden_dim
        self.select_idx = select_idx
        self.raw_text_prob = raw_text_prob
        if self.raw_text_prob is not None and not 0.0 <= float(self.raw_text_prob) <= 1.0:
            raise ValueError(f'raw_text_prob must be in [0, 1], got {raw_text_prob}')

    def _caption_path_to_t5_path(self, caption_path: str) -> str:
        """Map caption_path to the corresponding T5 feature .pt path.

        Logic: normalize caption_path, strip the data_dir parent prefix,
        replace 'motionhub/' (or data_dir basename) prefix if present,
        change .json -> .pt, prepend feature_dir.
        """
        full_path = os.path.normpath(caption_path)
        norm_data_dir = os.path.normpath(self.data_dir)
        data_parent = os.path.dirname(norm_data_dir)

        # Strip data_dir parent prefix to get relative path
        if full_path.startswith(data_parent + '/'):
            rel_path = full_path[len(data_parent) + 1:]
        elif full_path.startswith(data_parent):
            rel_path = full_path[len(data_parent):]
            if rel_path.startswith('/'):
                rel_path = rel_path[1:]
        else:
            # Fallback: use basename
            rel_path = os.path.basename(full_path)

        # Remove data_dir basename prefix (e.g. "motionhub/")
        data_dir_basename = os.path.basename(norm_data_dir)
        if rel_path.startswith(data_dir_basename + '/'):
            rel_path = rel_path[len(data_dir_basename) + 1:]

        # Motion caption files use a matching .pt feature-cache suffix.
        rel_path = os.path.splitext(rel_path)[0] + '.pt'

        return os.path.join(self.feature_dir, rel_path)

    def _t5_path_candidates(self, caption_path: str) -> List[str]:
        """Return current-layout and legacy-flat feature paths.

        Reprocessed TED/BEAT captions live one shard directory below
        ``hierarchical_caption`` while their pre-extracted T5 features retain
        the original flat layout. Prefer the path matching the source tree and
        only fall back to the flat cache when that file is absent.
        """
        primary = self._caption_path_to_t5_path(caption_path)
        candidates = [primary]
        parts = Path(primary).parts
        if 'hierarchical_caption' in parts:
            caption_idx = parts.index('hierarchical_caption')
            if len(parts) > caption_idx + 2:
                flat = str(Path(*parts[:caption_idx + 1], parts[-1]))
                if flat != primary:
                    shard = parts[caption_idx + 1]
                    if shard.startswith('part_') or shard.isdigit():
                        candidates = [flat, primary]
                    else:
                        candidates.append(flat)
        return candidates

    def transform(self, results: Dict) -> Optional[Dict]:
        caption_path = results.get('caption_path')
        if caption_path is None:
            if self.allow_none:
                return None  # Trigger refetch
            raise ValueError("LoadPreExtractedT5Feature: 'caption_path' not in results")

        pt_paths = self._t5_path_candidates(caption_path)
        pt_path = next((path for path in pt_paths if os.path.exists(path)), None)

        if pt_path is None:
            if self.allow_none:
                return None  # Trigger refetch — .pt not yet extracted
            raise FileNotFoundError(
                "LoadPreExtractedT5Feature: no feature path exists: "
                + ", ".join(pt_paths)
            )

        try:
            data = torch.load(pt_path, map_location='cpu', weights_only=False)
        except Exception as e:
            if self.allow_none:
                return None
            raise RuntimeError(f"Failed to load {pt_path}: {e}")

        embeddings = data.get('embeddings', [])
        captions = data.get('captions', [])
        seq_lens = data.get('seq_lens', [])
        source_types = data.get('source_types', [])

        if not embeddings:
            if self.allow_none:
                return None
            raise ValueError(f"No embeddings found in {pt_path}")

        if self.select_idx is None:
            if (
                self.raw_text_prob is not None
                and isinstance(source_types, list)
                and len(source_types) == len(embeddings)
            ):
                raw_indices = [
                    i for i, source_type in enumerate(source_types)
                    if source_type == 'raw'
                ]
                aug_indices = [
                    i for i, source_type in enumerate(source_types)
                    if source_type != 'raw'
                ]
                if raw_indices and aug_indices:
                    pool = raw_indices if random.random() < float(self.raw_text_prob) else aug_indices
                else:
                    pool = raw_indices or aug_indices
                idx = random.choice(pool)
            else:
                # Randomly select one variant (data augmentation, same as LoadCompatibleCaption).
                idx = random.randint(0, len(embeddings) - 1)
        else:
            idx = int(self.select_idx) % len(embeddings)
        emb = embeddings[idx]       # [seq_len_i, hidden_dim] bf16
        seq_len = seq_lens[idx]
        caption = captions[idx] if idx < len(captions) else ''

        # Truncate if longer than max_seq_length (shouldn't happen if extracted
        # with same max_seq_length, but defensive)
        if emb.size(0) > self.max_seq_length:
            emb = emb[:self.max_seq_length]
            seq_len = self.max_seq_length

        # Pad to max_seq_length with zeros (matches encode_prompt_with_mask)
        if emb.size(0) < self.max_seq_length:
            pad = torch.zeros(
                self.max_seq_length - emb.size(0), self.hidden_dim,
                dtype=emb.dtype
            )
            padded_emb = torch.cat([emb, pad], dim=0)
        else:
            padded_emb = emb

        # Build attention mask: 1 for valid tokens, 0 for padding
        mask = torch.zeros(self.max_seq_length, dtype=torch.long)
        mask[:seq_len] = 1

        results['t5_text_embeds'] = padded_emb       # [max_seq_length, 4096] bf16
        results['t5_text_mask'] = mask               # [max_seq_length] int64
        results['caption'] = caption                 # str for logging
        if isinstance(source_types, list) and idx < len(source_types):
            results['text_source_type'] = source_types[idx]

        return results


@TRANSFORMS.register_module(force=True)
class LoadHierarchicalCaption(BaseTransform):
    def __init__(
        self,
        key="caption",
        allow_none: bool = False,
        select_mode: str = "random",
    ):
        self.key = key
        self.allow_none = allow_none
        self.select_mode = select_mode
        assert self.select_mode in ["random", "first"]

    def transform(self, results: Dict) -> Dict:
        filename = results.get(f"{self.key}_path")
        if filename is None and self.allow_none:
            return results
        hierarchical_caption = read_json(filename)
        caption_list = []
        granularity_list = []
        for granularity in ["macro", "meso", "micro"]:
            assert (
                granularity in hierarchical_caption
            ), f"{filename} contains no {granularity} captions"
            captions = hierarchical_caption[granularity]
            for caption in captions:
                caption_list.append(caption)
                granularity_list.append(granularity)
        assert len(caption_list) > 0, f"{filename} contains no captions"
        select_idx = (
            0
            if self.select_mode == "first"
            else random.randint(0, len(caption_list) - 1)
        )
        results["caption"] = caption_list[select_idx]
        results["granularity"] = granularity_list[select_idx]
        results["caption_list"] = caption_list
        results["granularity_list"] = granularity_list
        return results


@TRANSFORMS.register_module(force=True)
class LoadHYMotionCaption(BaseTransform):
    def __init__(
        self,
        key="caption",
        allow_none: bool = False,
        select_mode: str = "random",
    ):
        self.key = key
        self.allow_none = allow_none
        self.select_mode = select_mode
        assert self.select_mode in ["random", "first"]

    def transform(self, results: Dict) -> Dict:
        filename = results.get(f"{self.key}_path")
        if filename is None and self.allow_none:
            return results
        hierarchical_caption = read_json(filename)
        caption_list = []
        granularity_list = []

        # 获取 result 数组
        result_list: List[Dict] = hierarchical_caption.get("result", [])

        # 遍历 result 数组中的每个元素
        # NOTE: Some caption files use "short caption" (space) instead of
        # "short_caption" (underscore). Accept both variants.
        for item in result_list:
            # 如果存在 short_caption_rewritten，使用它作为 caption 列表
            rewritten_key = (
                "short_caption_rewritten" if "short_caption_rewritten" in item
                else "short caption_rewritten" if "short caption_rewritten" in item
                else None
            )
            caption_key = (
                "short_caption" if "short_caption" in item
                else "short caption" if "short caption" in item
                else None
            )
            if rewritten_key is not None and isinstance(
                item[rewritten_key], list
            ):
                # short_caption_rewritten 是一个字符串数组
                for rewritten_caption in item[rewritten_key]:
                    if (
                        isinstance(rewritten_caption, str)
                        and len(rewritten_caption.strip()) > 0
                    ):
                        caption_list.append(rewritten_caption.strip())
            # 否则使用 short_caption
            elif caption_key is not None and isinstance(item[caption_key], str):
                short_caption = item[caption_key].strip()
                if len(short_caption) > 0:
                    caption_list.append(short_caption)

        assert len(caption_list) > 0, f"{filename} contains no captions"
        select_idx = random.randint(0, len(caption_list) - 1)
        results["caption"] = caption_list[select_idx]
        results["caption_list"] = caption_list
        return results


@TRANSFORMS.register_module(force=True)
class LoadCompatibleCaption(BaseTransform):
    """
    兼容两种 caption 格式的 transform：
    1. LoadHierarchicalCaption 格式：包含 "macro", "meso", "micro" 三个键
    2. LoadHYMotionCaption 格式：包含 "result" 数组
    如果两种格式都不符合，抛出异常。
    """

    def __init__(
        self,
        key="caption",
        allow_none: bool = False,
        select_mode: str = "random",
    ):
        self.key = key
        self.allow_none = allow_none
        self.select_mode = select_mode
        assert self.select_mode in ["random", "first"]

    def _is_hierarchical_format(self, data: Dict) -> bool:
        """判断是否为 LoadHierarchicalCaption 格式（包含 macro, meso, micro）"""
        required_keys = ["macro", "meso", "micro"]
        # 检查所有必需的键都存在且是列表（允许空列表，因为原始实现只检查存在性）
        return all(
            key in data and isinstance(data[key], list) for key in required_keys
        )

    def _is_hymotion_format(self, data: Dict) -> bool:
        """判断是否为 LoadHYMotionCaption 格式（包含 result 数组）"""
        if "result" not in data:
            return False
        result_list = data["result"]
        if not isinstance(result_list, list) or len(result_list) == 0:
            return False
        # 检查 result 数组中的元素是否有 short_caption 或 short_caption_rewritten
        # Also accept "short caption" (space) variant
        for item in result_list:
            if not isinstance(item, dict):
                continue
            if any(k in item for k in ("short_caption", "short_caption_rewritten",
                                        "short caption", "short caption_rewritten")):
                return True
        return False

    def _select_caption_from_file(self, filename: str):
        hierarchical_caption = read_json(filename)
        caption_list = []
        granularity_list = []

        # 判断格式并处理
        if self._is_hierarchical_format(hierarchical_caption):
            # LoadHierarchicalCaption 格式
            for granularity in ["macro", "meso", "micro"]:
                captions = hierarchical_caption[granularity]
                for caption in captions:
                    caption_list.append(caption)
                    granularity_list.append(granularity)
            assert len(caption_list) > 0, f"{filename} contains no captions"

        elif self._is_hymotion_format(hierarchical_caption):
            # LoadHYMotionCaption 格式
            result_list: List[Dict] = hierarchical_caption.get("result", [])
            for item in result_list:
                rewritten_key = (
                    "short_caption_rewritten" if "short_caption_rewritten" in item
                    else "short caption_rewritten" if "short caption_rewritten" in item
                    else None
                )
                caption_key = (
                    "short_caption" if "short_caption" in item
                    else "short caption" if "short caption" in item
                    else None
                )
                if rewritten_key is not None and isinstance(
                    item[rewritten_key], list
                ):
                    for rewritten_caption in item[rewritten_key]:
                        if (
                            isinstance(rewritten_caption, str)
                            and len(rewritten_caption.strip()) > 0
                        ):
                            caption_list.append(rewritten_caption.strip())
                elif caption_key is not None and isinstance(item[caption_key], str):
                    short_caption = item[caption_key].strip()
                    if len(short_caption) > 0:
                        caption_list.append(short_caption)
            assert len(caption_list) > 0, f"{filename} contains no captions"

        else:
            # 两种格式都不符合，抛出异常
            raise ValueError(
                f"{filename} does not match either format:\n"
                f"  - LoadHierarchicalCaption: requires 'macro', 'meso', 'micro' keys\n"
                f"  - LoadHYMotionCaption: requires 'result' array with 'short_caption' or 'short_caption_rewritten'"
            )

        select_idx = (
            0
            if self.select_mode == "first"
            else random.randint(0, len(caption_list) - 1)
        )
        granularity = (
            granularity_list[select_idx]
            if select_idx < len(granularity_list)
            else None
        )
        return caption_list[select_idx], caption_list, granularity, granularity_list

    def transform(self, results: Dict) -> Dict:
        inline_caption_list = results.get(f"{self.key}_list")
        if isinstance(inline_caption_list, (list, tuple)):
            caption_list = [
                caption.strip()
                for caption in inline_caption_list
                if isinstance(caption, str) and caption.strip()
            ]
            if caption_list:
                select_idx = (
                    0
                    if self.select_mode == "first"
                    else random.randint(0, len(caption_list) - 1)
                )
                results[self.key] = caption_list[select_idx]
                results[f"{self.key}_list"] = caption_list
                return results

        inline_caption = results.get(self.key)
        if isinstance(inline_caption, str) and inline_caption.strip():
            caption = inline_caption.strip()
            results[self.key] = caption
            results[f"{self.key}_list"] = [caption]
            return results

        filename = results.get(f"{self.key}_path")
        if filename is None and self.allow_none:
            return results

        caption, caption_list, granularity, granularity_list = (
            self._select_caption_from_file(filename)
        )
        results["caption"] = caption
        results["caption_list"] = caption_list
        if granularity is not None:
            results["granularity"] = granularity
            results["granularity_list"] = granularity_list

        person_caption_paths = results.get("person_caption_paths")
        if isinstance(person_caption_paths, (list, tuple)) and person_caption_paths:
            person_captions = []
            person_caption_lists = []
            for person_caption_path in person_caption_paths:
                if person_caption_path is None:
                    continue
                person_caption, person_caption_list, _, _ = (
                    self._select_caption_from_file(person_caption_path)
                )
                person_captions.append(person_caption)
                person_caption_lists.append(person_caption_list)
            if len(person_captions) == len(person_caption_paths):
                results["person_captions"] = person_captions
                results["person_caption_lists"] = person_caption_lists

        return results


@TRANSFORMS.register_module(force=True)
class LoadHm3dTxt(BaseTransform):

    def __init__(
        self, keys: Union[str, List[str]] = "caption", min_duration=0, sr=None
    ):
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

        self.sr = sr
        self.min_duration = min_duration

    def transform(self, results: dict) -> dict:
        """Functions to load humanml3d caption text.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.
        Returns:
            dict: The dict contains loaded caption, token, etc.
        """
        for key in self.keys:
            filename = results.get(f"{key}_path")
            if filename is None or not os.path.exists(filename):
                continue

            caption_list, pos_list, range_list = self.load_caption(filename)
            # 0 <= idx <= num_captions - 1
            select_idx = random.randint(0, len(caption_list) - 1)
            caption = caption_list[select_idx]
            pos = pos_list[select_idx]
            range = range_list[select_idx]

            results[key] = caption
            results[f"{key}_pos"] = pos
            results[f"{key}_range"] = range

            results[f"{key}_list"] = caption_list
            # pos: part of speech
            results[f"{key}_pos_list"] = pos_list
            results[f"{key}_range_list"] = range_list

        return results

    @staticmethod
    def judge_hm3d(content: str):
        """Judge if the content is a humanml3d type caption file
        :param content: content of file
        :return: True or False
        """
        content = content.strip()

        first_line = content.split("\n")[0]
        if hm3d_pattern.match(first_line):
            return True
        return False

    def load_hm3d_caption(self, content: str):
        caption_list = []
        pos_list = []
        range_list = []

        for line in content.split("\n"):
            caption = line.split("#")[0].strip()
            assert len(caption) > 0, content
            pos = line.split("#")[1].strip()

            range = line.split("#")[-2:]
            range = [float(x) for x in range]
            duration = range[1] - range[0]
            # duration == 0 means no crop occurs.
            if 0 < duration < self.min_duration:
                continue

            caption_list.append(caption)
            pos_list.append(pos)
            range_list.append(range)
        return caption_list, pos_list, range_list

    @staticmethod
    def load_pure_caption(content: str):
        caption_list = []
        pos_list = []
        range_list = []
        for line in content.split("\n"):
            caption = line.strip()

            caption_list.append(caption)

            pos_list.append(None)
            range_list.append([0, 0])
        return caption_list, pos_list, range_list

    def load_caption(self, caption_path: str) -> Tuple:
        """
        :param caption_path: txt path of humanml3d caption file.
        :return: caption list, pos list and range list
        """
        try:
            content = read_txt(caption_path).strip()
        except:
            raise Exception(caption_path)
        is_hm3d = self.judge_hm3d(content)
        if is_hm3d:
            caption_list, pos_list, range_list = self.load_hm3d_caption(content)
        else:
            caption_list, pos_list, range_list = self.load_pure_caption(content)

        return caption_list, pos_list, range_list

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(" f"key={self.key})"

        return repr_str


@TRANSFORMS.register_module(force=True)
class LoadTxt(BaseTransform):
    def __init__(self, key: str = "speech_script", allow_none: bool = False):
        self.key = key
        self.allow_none = allow_none

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        filename = results.get(f"{self.key}_path")
        if filename is None and self.allow_none:
            return results

        text = read_txt(filename)

        results[self.key] = text
        return results
