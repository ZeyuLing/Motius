"""BABEL action-taxonomy helpers for sequential retrieval evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence


def normalize_action_text(value: object) -> str:
    """Normalize a BABEL label while retaining its lexical meaning."""

    return " ".join(re.findall(r"\w+", str(value or "").casefold(), flags=re.UNICODE))


def build_action_group(
    categories: Sequence[Sequence[str]],
    fallback_labels: Sequence[str],
) -> tuple[str, list[dict[str, object]]]:
    """Build a stable ID from an ordered sequence of BABEL action categories."""

    if len(categories) != len(fallback_labels):
        raise ValueError("Action categories and fallback labels must have equal length.")
    signature: list[dict[str, object]] = []
    for values, fallback in zip(categories, fallback_labels):
        normalized = sorted(
            {normalize_action_text(value) for value in values if normalize_action_text(value)}
        )
        if normalized:
            signature.append({"act_cat": normalized})
            continue
        label = normalize_action_text(fallback)
        if not label:
            raise ValueError("BABEL action groups require a category or fallback label.")
        signature.append({"proc_label": label})
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]
    return f"babel-act-cat-v1:{digest}", signature


@dataclass(frozen=True)
class BabelActionEntry:
    """One resolved action category and its provenance."""

    categories: tuple[str, ...]
    source: str


class BabelActionCatalog:
    """Resolve raw and processed BABEL labels to official ``act_cat`` values."""

    def __init__(self, annotations: Mapping[str, object]):
        raw_entries: dict[str, BabelActionEntry] = {}
        proc_entries: dict[str, BabelActionEntry] = {}
        sequence_entries: dict[str, tuple[str, ...]] = {}

        def add_unique(
            destination: dict[str, BabelActionEntry],
            label: object,
            categories: Sequence[object] | None,
            source: str,
        ) -> None:
            key = normalize_action_text(label)
            if not key:
                return
            entry = BabelActionEntry(
                tuple(
                    sorted(
                        {
                            normalize_action_text(value)
                            for value in (categories or [])
                            if normalize_action_text(value)
                        }
                    )
                ),
                source,
            )
            previous = destination.get(key)
            if previous is not None and previous.categories != entry.categories:
                raise ValueError(
                    f"Ambiguous BABEL action taxonomy for {label!r}: "
                    f"{previous.categories} vs {entry.categories}."
                )
            destination[key] = entry

        for babel_id, annotation_value in annotations.items():
            annotation = dict(annotation_value)
            sequence_categories: set[str] = set()
            for segment in ((annotation.get("seq_ann") or {}).get("labels") or []):
                sequence_categories.update(
                    normalize_action_text(value)
                    for value in (segment.get("act_cat") or [])
                    if normalize_action_text(value)
                )
            sequence_entries[str(babel_id)] = tuple(sorted(sequence_categories))
            for segment in ((annotation.get("frame_ann") or {}).get("labels") or []):
                categories = segment.get("act_cat") or []
                add_unique(
                    raw_entries,
                    segment.get("raw_label"),
                    categories,
                    "official_frame_raw_label",
                )
                add_unique(
                    proc_entries,
                    segment.get("proc_label"),
                    categories,
                    "official_frame_proc_label",
                )

        self._raw_entries = raw_entries
        self._proc_entries = proc_entries
        self._sequence_entries = sequence_entries

    def resolve(
        self,
        label: str,
        *,
        babel_id: str,
        processed: bool,
    ) -> BabelActionEntry:
        entries = self._proc_entries if processed else self._raw_entries
        entry = entries.get(normalize_action_text(label))
        if entry is not None:
            return entry
        sequence_categories = self._sequence_entries.get(str(babel_id), ())
        if sequence_categories:
            return BabelActionEntry(
                sequence_categories, "official_sequence_act_cat_fallback"
            )
        return BabelActionEntry((), "normalized_label_fallback")


def enrich_manifest_action_groups(
    manifest: Mapping[str, object],
    annotations: Mapping[str, object],
    *,
    protocol: str,
) -> tuple[dict[str, object], dict[str, int]]:
    """Attach official action-category positive groups to a BABEL manifest."""

    output = copy.deepcopy(dict(manifest))
    catalog = BabelActionCatalog(annotations)
    stats: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    for case in output.get("cases", []):
        babel_id = str(case.get("babel_id") or "")
        merge_notes = {
            (int(note["start"]), int(note["end"])): note
            for note in case.get("protocol_notes", [])
            if note.get("kind") == "short_action_merge"
        }
        for segment in case.get("segments", []):
            span = (int(segment["start_frame"]), int(segment["end_frame"]))
            if segment.get("merged"):
                note = merge_notes.get(span)
                if note is None:
                    raise ValueError(
                        f"Merged BABEL segment {case.get('case_id')} {span} has no source note."
                    )
                labels = [str(value) for value in note.get("source_captions", [])]
                processed = True
            else:
                labels = [str(segment.get("raw_label") or segment.get("caption") or "")]
                processed = False
            entries = [
                catalog.resolve(label, babel_id=babel_id, processed=processed)
                for label in labels
            ]
            categories = [list(entry.categories) for entry in entries]
            group_id, signature = build_action_group(categories, labels)
            segment["action_group_id"] = group_id
            segment["action_group_signature"] = signature
            segment["action_categories"] = categories
            segment["action_source_labels"] = labels
            for entry in entries:
                stats[entry.source] += 1
            stats["source_actions"] += len(entries)
            stats["segments"] += 1
            group_counts[group_id] += 1

    output["protocol"] = str(protocol)
    output["retrieval_positive_policy"] = "official BABEL act_cat ordered-signature multi-positive"
    output["action_group_policy"] = {
        "taxonomy": "official BABEL act_cat",
        "signature": "ordered source actions; sorted categories within each action",
        "fallback": "proc_label when act_cat is unavailable",
        "id_version": "babel-act-cat-v1",
        "unique_groups": len(group_counts),
        "duplicate_groups": sum(count > 1 for count in group_counts.values()),
        "segments_in_duplicate_groups": sum(
            count for count in group_counts.values() if count > 1
        ),
    }
    stats["unique_groups"] = len(group_counts)
    stats["duplicate_groups"] = sum(count > 1 for count in group_counts.values())
    stats["segments_in_duplicate_groups"] = sum(
        count for count in group_counts.values() if count > 1
    )
    return output, dict(stats)


def positive_group_id(segment: Mapping[str, object]) -> str:
    """Return an action group, falling back to exact-caption grouping."""

    value = str(segment.get("action_group_id") or "").strip()
    if value:
        return value
    return f"caption-v1:{normalize_action_text(segment.get('caption'))}"


__all__ = [
    "BabelActionCatalog",
    "BabelActionEntry",
    "build_action_group",
    "enrich_manifest_action_groups",
    "normalize_action_text",
    "positive_group_id",
]
