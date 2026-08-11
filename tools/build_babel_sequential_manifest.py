#!/usr/bin/env python3
"""Build the Motius BABEL sequential-generation benchmark.

The benchmark is the union of eligible episodes in the official BABEL
validation split. Explicit transition labels are removed and split at their
midpoint between neighboring actions. Adjacent action intervals are then
greedily merged until every text-conditioned segment is at least 30 frames.
Captions come from a precomputed LLM rewrite cache.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.evaluation.babel import enrich_manifest_action_groups
from motius.motion import canonicalize_smpl22_joints, motion272_to_joints
from motius.motion.skeleton import smpl22_rest_offsets


PROTOCOL = "babel-official-val-shortmerge30-llm-joints66-actiongroups-v3"


def _is_transition(segment: Mapping[str, object]) -> bool:
    label = str(
        segment.get("caption")
        or segment.get("proc_label")
        or segment.get("raw_label")
        or ""
    )
    return bool(segment.get("is_transition")) or label.strip().lower() == "transition"


def _frame_at_time(seconds: object, fps: float) -> int:
    return int(round(float(seconds) * float(fps)))


def _to_target_frames(
    segment: Mapping[str, object],
    *,
    target_fps: float,
    total_frames: int,
) -> dict[str, object] | None:
    if "start_t" in segment and "end_t" in segment:
        start = _frame_at_time(segment["start_t"], target_fps)
        end = _frame_at_time(segment["end_t"], target_fps)
    else:
        source_fps = float(segment.get("fps") or target_fps)
        start = int(round(float(segment["start_frame"]) * target_fps / source_fps))
        end = int(round(float(segment["end_frame"]) * target_fps / source_fps))
    start = max(0, min(start, total_frames))
    end = max(0, min(end, total_frames))
    caption = str(
        segment.get("caption")
        or segment.get("proc_label")
        or segment.get("raw_label")
        or ""
    ).strip()
    if end <= start or not caption:
        return None
    return {
        "caption": caption,
        "raw_label": str(segment.get("raw_label") or "").strip(),
        "start": start,
        "end": end,
        "is_transition": _is_transition(segment),
    }


def _transition_midpoint(
    left: Mapping[str, object],
    right: Mapping[str, object],
    transitions: Iterable[Mapping[str, object]],
) -> tuple[int | None, int]:
    hits = []
    for transition in transitions:
        if int(transition["end"]) <= int(left["start"]):
            continue
        if int(transition["start"]) >= int(right["end"]):
            continue
        if (
            int(transition["start"]) >= int(left["end"]) - 1
            and int(transition["end"]) <= int(right["start"]) + 1
        ):
            hits.append(transition)
    if not hits:
        return None, 0
    start = min(int(item["start"]) for item in hits)
    end = max(int(item["end"]) for item in hits)
    return int(round((start + end) / 2.0)), len(hits)


def build_official_episode(
    raw: Mapping[str, object],
    *,
    target_fps: float = 30.0,
    min_segments: int = 2,
) -> tuple[dict[str, object] | None, Counter]:
    """Remove transition labels and construct contiguous action intervals."""

    stats: Counter = Counter()
    duration = raw.get("duration_sec_npz") or raw.get("duration_sec_babel")
    if duration is None:
        duration = float(raw["num_frames"]) / float(raw["fps"])
    total_frames = max(1, _frame_at_time(duration, target_fps))
    segments = []
    for segment in raw.get("segments", []):
        if segment.get("seq_level"):
            continue
        converted = _to_target_frames(
            segment,
            target_fps=target_fps,
            total_frames=total_frames,
        )
        if converted is not None:
            segments.append(converted)
    segments.sort(key=lambda item: (item["start"], item["end"], item["caption"]))
    actions = [item for item in segments if not item["is_transition"]]
    transitions = [item for item in segments if item["is_transition"]]
    if len(actions) < min_segments:
        stats["skipped_few_actions"] += 1
        return None, stats

    starts = [int(actions[0]["start"])]
    ends = []
    notes = []
    for left, right in zip(actions, actions[1:]):
        midpoint, transition_count = _transition_midpoint(left, right, transitions)
        if midpoint is not None:
            cut = midpoint
            stats["explicit_transition_cuts"] += 1
            notes.append(
                {
                    "kind": "transition_midpoint",
                    "between": [left["caption"], right["caption"]],
                    "cut": cut,
                    "transition_count": transition_count,
                }
            )
        else:
            cut = int(right["start"])
            stats["native_cuts"] += 1
            if int(left["end"]) != int(right["start"]):
                notes.append(
                    {
                        "kind": "native_onset_overlap_or_gap",
                        "between": [left["caption"], right["caption"]],
                        "left_end": int(left["end"]),
                        "right_start": int(right["start"]),
                        "cut": cut,
                    }
                )
        cut = max(starts[-1] + 1, min(cut, int(right["end"]) - 1))
        ends.append(cut)
        starts.append(cut)
    ends.append(int(actions[-1]["end"]))

    clip_start = starts[0]
    clip_end = ends[-1]
    if clip_end <= clip_start:
        stats["skipped_bad_span"] += 1
        return None, stats
    output_segments = []
    for action, start, end in zip(actions, starts, ends):
        relative_start = int(start - clip_start)
        relative_end = int(end - clip_start)
        if relative_end <= relative_start:
            continue
        output_segments.append(
            {
                "caption": str(action["caption"]),
                "raw_label": str(action["raw_label"]),
                "start": relative_start,
                "end": relative_end,
                "source_start_30": int(start),
                "source_end_30": int(end),
            }
        )
    if len(output_segments) < min_segments:
        stats["skipped_after_empty"] += 1
        return None, stats
    stats["episodes"] += 1
    stats["action_segments"] += len(output_segments)
    return (
        {
            "id": str(raw["id"]),
            "babel_id": raw.get("babel_id"),
            "split": str(raw.get("split") or "val"),
            "amass_path": raw.get("amass_path"),
            "target_fps": float(target_fps),
            "source_start_30": int(clip_start),
            "source_end_30": int(clip_end),
            "total_frames": int(clip_end - clip_start),
            "segments": output_segments,
            "protocol_notes": notes,
        },
        stats,
    )


def _segment_length(segment: Mapping[str, object]) -> int:
    return max(0, int(segment["end"]) - int(segment["start"]))


def _fallback_caption(group: list[Mapping[str, object]]) -> str:
    pieces = [
        str(item.get("caption") or item.get("raw_label") or "").strip().rstrip(".")
        for item in group
    ]
    pieces = [piece for piece in pieces if piece]
    if not pieces:
        return "A person moves."
    body = ", then ".join(pieces)
    body = re.sub(r"^(a|the) person\s+", "", body, flags=re.IGNORECASE)
    return f"A person {body}."


def merge_short_segments(
    episode: Mapping[str, object],
    *,
    min_frames: int = 30,
) -> tuple[dict[str, object], Counter]:
    """Greedily merge short action intervals with their following neighbor."""

    source_segments = list(episode.get("segments") or [])
    groups: list[list[Mapping[str, object]]] = []
    stats: Counter = Counter()
    index = 0
    while index < len(source_segments):
        group = [source_segments[index]]
        frames = _segment_length(source_segments[index])
        index += 1
        if frames < min_frames:
            stats["short_seed_segments"] += 1
            while frames < min_frames and index < len(source_segments):
                group.append(source_segments[index])
                frames += _segment_length(source_segments[index])
                index += 1
        groups.append(group)
    if len(groups) > 1 and sum(_segment_length(item) for item in groups[-1]) < min_frames:
        groups[-2].extend(groups.pop())
        stats["merged_short_tail"] += 1

    merged_segments = []
    notes = list(episode.get("protocol_notes") or [])
    for group in groups:
        source = []
        for item in group:
            source.append(
                {
                    "caption": str(item.get("caption") or ""),
                    "raw_label": str(item.get("raw_label") or ""),
                    "start": int(item["start"]),
                    "end": int(item["end"]),
                    "frames": _segment_length(item),
                    "source_start_30": int(item.get("source_start_30", item["start"])),
                    "source_end_30": int(item.get("source_end_30", item["end"])),
                }
            )
        merged = len(group) > 1
        if merged:
            stats["merged_groups"] += 1
            stats["merged_source_segments"] += len(group)
            notes.append(
                {
                    "kind": "short_action_merge",
                    "min_frames": int(min_frames),
                    "start": int(group[0]["start"]),
                    "end": int(group[-1]["end"]),
                    "source_captions": [item["caption"] for item in source],
                    "source_lengths": [item["frames"] for item in source],
                }
            )
        merged_segments.append(
            {
                "caption": _fallback_caption(group),
                "raw_label": " then ".join(
                    str(item.get("raw_label") or item.get("caption") or "").strip()
                    for item in group
                ),
                "start": int(group[0]["start"]),
                "end": int(group[-1]["end"]),
                "source_start_30": int(group[0].get("source_start_30", group[0]["start"])),
                "source_end_30": int(group[-1].get("source_end_30", group[-1]["end"])),
                "merged": merged,
                "merged_source_segments": source,
            }
        )
    stats["original_segments"] += len(source_segments)
    stats["merged_segments"] += len(merged_segments)
    stats["remaining_short_segments"] += sum(
        _segment_length(item) < min_frames for item in merged_segments
    )
    output = dict(episode)
    output["segments"] = merged_segments
    output["protocol_notes"] = notes
    output["shortmerge"] = {
        "min_frames": int(min_frames),
        "old_num_segments": len(source_segments),
        "new_num_segments": len(merged_segments),
        "num_merged_groups": sum(bool(item["merged"]) for item in merged_segments),
    }
    return output, stats


def _rewrite_payload(segment: Mapping[str, object], mode: str) -> str:
    source = list(segment.get("merged_source_segments") or [])
    normalized = [
        re.sub(
            r"\s+",
            " ",
            str(item.get("caption") or item.get("raw_label") or "").strip(),
        )
        for item in source
    ]
    if mode == "labels":
        return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    if mode == "span":
        return "||".join(
            f"{int(item.get('start', 0))}:{int(item.get('end', 0))}:"
            f"{caption}"
            for item, caption in zip(source, normalized)
        )
    raise ValueError(f"Unsupported rewrite key mode: {mode}")


def _rewrite_key(segment: Mapping[str, object], mode: str) -> str:
    payload = _rewrite_payload(segment, mode)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{mode}:{digest}:{payload}"


def load_rewrite_cache(path: str | Path) -> dict[str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    values = data.get("rewrites", data)
    return {str(key): str(value).strip() for key, value in values.items()}


def apply_rewrites(
    episode: Mapping[str, object],
    rewrites: Mapping[str, str],
    *,
    require_rewrite: bool = True,
) -> tuple[dict[str, object], Counter]:
    """Apply LLM captions, accepting both label- and span-keyed caches."""

    output = dict(episode)
    segments = []
    stats: Counter = Counter()
    for raw_segment in episode.get("segments") or []:
        segment = dict(raw_segment)
        caption = None
        key_used = None
        for mode in ("labels", "span"):
            key = _rewrite_key(segment, mode)
            if key in rewrites:
                caption = str(rewrites[key]).strip()
                key_used = key
                break
        if not caption:
            stats["rewrite_misses"] += 1
            if require_rewrite:
                labels = [
                    item.get("caption") or item.get("raw_label")
                    for item in segment.get("merged_source_segments") or []
                ]
                raise KeyError(f"Missing LLM caption rewrite for {episode.get('id')}: {labels}")
            caption = str(segment["caption"])
            segment["caption_source"] = "rule_fallback"
        else:
            stats["rewrite_hits"] += 1
            segment["caption_rule_before_llm"] = segment["caption"]
            segment["caption_source"] = "llm_shortmerge_rewrite"
            segment["caption_rewrite_key"] = key_used
        segment["caption"] = caption
        segments.append(segment)
    output["segments"] = segments
    return output, stats


def _motion_path(motion_dir: Path, case_id: str) -> Path:
    for suffix in (".npz", ".npy"):
        candidate = motion_dir / f"{case_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No MS272 GT file for {case_id} under {motion_dir}")


def _load_motion272(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        if "motion_272" not in value:
            raise ValueError(f"{path} does not contain motion_272")
        motion = value["motion_272"]
        value.close()
    else:
        motion = value
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 272 or not np.isfinite(motion).all():
        raise ValueError(f"Invalid MS272 motion at {path}: {motion.shape}")
    return motion


def _public_segment(segment: Mapping[str, object]) -> dict[str, object]:
    return {
        "caption": str(segment["caption"]),
        "start_frame": int(segment["start"]),
        "end_frame": int(segment["end"]),
        "raw_label": str(segment.get("raw_label") or ""),
        "merged": bool(segment.get("merged")),
        "caption_source": str(segment.get("caption_source") or ""),
    }


def build_protocol(
    records: Iterable[Mapping[str, object]],
    *,
    motion272_dir: str | Path,
    smpl22_offsets: np.ndarray,
    rewrite_cache: Mapping[str, str],
    babel_annotations: Mapping[str, object],
    output_root: str | Path,
    target_fps: float = 30.0,
    min_frames: int = 30,
    min_segments: int = 2,
    limit: int = 0,
    workers: int = 1,
) -> dict[str, object]:
    output_root = Path(output_root).resolve()
    offsets = np.asarray(smpl22_offsets, dtype=np.float32)
    if offsets.shape != (22, 3) or not np.isfinite(offsets).all():
        raise ValueError(
            f"smpl22_offsets must be finite with shape (22,3), got {offsets.shape}"
        )
    pelvis_offsets = offsets.copy()
    pelvis_offsets[0] = 0.0
    offsets_path = output_root / "smpl22_offsets_y.npy"
    output_root.mkdir(parents=True, exist_ok=True)
    np.save(offsets_path, offsets)
    reference_dir = output_root / "references" / "joints66"
    reference_dir.mkdir(parents=True, exist_ok=True)
    motion272_dir = Path(motion272_dir).resolve()
    episodes = []
    stats: Counter = Counter()
    for index, raw in enumerate(records):
        if limit and index >= limit:
            break
        episode, episode_stats = build_official_episode(
            raw,
            target_fps=target_fps,
            min_segments=min_segments,
        )
        stats.update(episode_stats)
        if episode is None:
            continue
        episode, merge_stats = merge_short_segments(episode, min_frames=min_frames)
        stats.update(merge_stats)
        episode, rewrite_stats = apply_rewrites(episode, rewrite_cache)
        stats.update(rewrite_stats)
        episodes.append(episode)

    def materialize(episode: Mapping[str, object]):
        source = _load_motion272(_motion_path(motion272_dir, str(episode["id"])))
        source_start = int(episode["source_start_30"])
        source_end = int(episode["source_end_30"])
        episode_frames = int(episode["total_frames"])
        if len(source) == episode_frames:
            episode_motion = source
            source_kind = "preclipped_motion_files"
        elif source_end <= len(source):
            episode_motion = source[source_start:source_end]
            source_kind = "full_motion_files"
        else:
            raise ValueError(
                f"{episode['id']} needs source frames [{source_start}, {source_end}), "
                f"but its MS272 file contains {len(source)} frames"
            )
        joints = motion272_to_joints(episode_motion, bone_offsets=pelvis_offsets)
        joints = canonicalize_smpl22_joints(joints).reshape(len(joints), 66)
        reference_path = reference_dir / f"{episode['id']}.npy"
        np.save(reference_path, joints)
        return (
            {
                "case_id": str(episode["id"]),
                "babel_id": episode.get("babel_id"),
                "amass_path": episode.get("amass_path"),
                "total_frames": int(episode["total_frames"]),
                "reference_path": reference_path.relative_to(output_root).as_posix(),
                "segments": [_public_segment(item) for item in episode["segments"]],
                "protocol_notes": episode.get("protocol_notes", []),
            },
            source_kind,
        )

    worker_count = max(1, int(workers))
    if worker_count == 1:
        materialized = map(materialize, episodes)
    else:
        executor = ThreadPoolExecutor(max_workers=worker_count)
        materialized = executor.map(materialize, episodes)
    cases = []
    try:
        for case, source_kind in materialized:
            cases.append(case)
            stats[source_kind] += 1
    finally:
        if worker_count > 1:
            executor.shutdown(wait=True)
    transitions = sum(max(0, len(case["segments"]) - 1) for case in cases)
    manifest = {
        "protocol": PROTOCOL,
        "split": "val",
        "fps": float(target_fps),
        "motion_representation": "canonical SMPL-22 joints66",
        "data_source": "official BABEL validation episodes",
        "episode_policy": "all episodes with at least two non-transition actions",
        "boundary_policy": "explicit transition midpoint; otherwise next action onset",
        "short_action_policy": f"greedy adjacent merge to at least {int(min_frames)} frames",
        "caption_policy": "LLM rewrite of every merged source-label sequence",
        "skeleton_policy": "neutral zero-beta SMPL-22 for references and predictions",
        "smpl22_offsets": offsets_path.relative_to(output_root).as_posix(),
        "counts": {
            "episodes": len(cases),
            "original_action_segments": int(stats["original_segments"]),
            "captioned_segments": int(sum(len(case["segments"]) for case in cases)),
            "transition_boundaries": int(transitions),
            "merged_groups": int(stats["merged_groups"]),
            "rewrite_hits": int(stats["rewrite_hits"]),
            "rewrite_misses": int(stats["rewrite_misses"]),
        },
        "cases": cases,
    }
    manifest, action_group_stats = enrich_manifest_action_groups(
        manifest,
        babel_annotations,
        protocol=PROTOCOL,
    )
    manifest["counts"]["action_groups"] = int(action_group_stats["unique_groups"])
    stats.update(
        {f"action_group_{key}": value for key, value in action_group_stats.items()}
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "build_stats.json").write_text(
        json.dumps(dict(stats), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _read_jsonl(path: str | Path) -> Iterable[dict[str, object]]:
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-manifest",
        default="data/babel/processed/manifests/val.jsonl",
        help="Processed official BABEL val JSONL with action and transition spans.",
    )
    parser.add_argument(
        "--motion272-dir",
        default="data/babel/processed/ms272/val",
        help="Directory containing one {episode_id}.npz motion_272 file per episode.",
    )
    parser.add_argument(
        "--rewrite-cache",
        default="data/babel/processed/babel_shortmerge_caption_rewrites.json",
        help="Precomputed label-sequence to LLM-caption cache.",
    )
    parser.add_argument(
        "--babel-annotations",
        default="data/babel/babel-teach/val.json",
        help="Official BABEL val.json containing proc_label and act_cat fields.",
    )
    parser.add_argument(
        "--smpl-model",
        default="checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl",
        help="Licensed neutral SMPL model file or directory used to standardize the skeleton.",
    )
    parser.add_argument(
        "--smpl-model-type",
        default="smpl",
        choices=("smpl", "smplh", "smplx"),
    )
    parser.add_argument(
        "--smpl-gender",
        default="neutral",
        choices=("neutral", "male", "female"),
    )
    parser.add_argument(
        "--output-root",
        default="outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1",
    )
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--min-segments", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_protocol(
        _read_jsonl(args.processed_manifest),
        motion272_dir=args.motion272_dir,
        smpl22_offsets=smpl22_rest_offsets(
            args.smpl_model,
            model_type=args.smpl_model_type,
            gender=args.smpl_gender,
            root_origin="model",
        ),
        rewrite_cache=load_rewrite_cache(args.rewrite_cache),
        babel_annotations=json.loads(
            Path(args.babel_annotations).read_text(encoding="utf-8")
        ),
        output_root=args.output_root,
        target_fps=args.target_fps,
        min_frames=args.min_frames,
        min_segments=args.min_segments,
        limit=args.limit,
        workers=args.workers,
    )
    print(json.dumps({"protocol": manifest["protocol"], **manifest["counts"]}, indent=2))
    print(f"manifest: {Path(args.output_root).resolve() / 'manifest.json'}")


if __name__ == "__main__":
    main()
