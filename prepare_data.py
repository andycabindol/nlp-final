"""Data preparation utilities for Phase 1 NER benchmarking.

Loads supported datasets, reconstructs raw text, and converts BIO tags
to span entities with character offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings


Entity = Dict[str, object]
Example = Dict[str, object]


@dataclass
class DatasetSpec:
    name: str
    subset: Optional[str]
    split: str


def _safe_import_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "HuggingFace 'datasets' is required. Install with: pip install datasets"
        ) from exc
    return load_dataset


def _load_with_compat(load_dataset, name: str, subset: Optional[str] = None, split: str = "validation"):
    """Compatibility loader across datasets versions."""
    kwargs = {"split": split}
    if subset is not None:
        args = (name, subset)
    else:
        args = (name,)
    try:
        return load_dataset(*args, trust_remote_code=True, **kwargs)
    except TypeError:
        # Older/newer versions may not expose this flag.
        return load_dataset(*args, **kwargs)
    except RuntimeError as exc:
        # Fallback for environments where remote script trust is blocked.
        if "Dataset scripts are no longer supported" in str(exc):
            return load_dataset(*args, **kwargs)
        raise


def _normalize_token(token: str) -> str:
    # CoNLL-style placeholders that sometimes appear in tokenized corpora.
    replacements = {
        "-LRB-": "(",
        "-RRB-": ")",
        "-LSB-": "[",
        "-RSB-": "]",
        "-LCB-": "{",
        "-RCB-": "}",
        "``": '"',
        "''": '"',
    }
    return replacements.get(token, token)


def reconstruct_text_and_offsets(tokens: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """Rebuild sentence text and track char offsets for each token."""
    text_parts: List[str] = []
    offsets: List[Tuple[int, int]] = []
    cursor = 0

    no_space_before = {".", ",", "!", "?", ":", ";", "%", ")", "]", "}", "'s"}
    no_space_after = {"(", "[", "{", "$", "#"}

    prev_tok: Optional[str] = None
    for raw_tok in tokens:
        tok = _normalize_token(raw_tok)

        add_space = bool(text_parts)
        if tok in no_space_before:
            add_space = False
        if prev_tok in no_space_after:
            add_space = False
        if tok.startswith("'") and len(tok) > 1:
            add_space = False

        if add_space:
            text_parts.append(" ")
            cursor += 1

        start = cursor
        text_parts.append(tok)
        cursor += len(tok)
        end = cursor

        offsets.append((start, end))
        prev_tok = tok

    return "".join(text_parts), offsets


def bio_to_entities(
    tokens: List[str], offsets: List[Tuple[int, int]], tag_names: List[str]
) -> List[Entity]:
    """Convert BIO token tags into span entities with char offsets."""
    entities: List[Entity] = []

    active_label: Optional[str] = None
    active_start_token: Optional[int] = None
    active_end_token: Optional[int] = None

    def flush_active():
        nonlocal active_label, active_start_token, active_end_token
        if active_label is None or active_start_token is None or active_end_token is None:
            return
        start_char = offsets[active_start_token][0]
        end_char = offsets[active_end_token][1]
        text_slice = text[start_char:end_char]
        entities.append(
            {
                "text": text_slice,
                "label": active_label,
                "start": start_char,
                "end": end_char,
            }
        )
        active_label = None
        active_start_token = None
        active_end_token = None

    text, _ = reconstruct_text_and_offsets(tokens)
    for i, tag in enumerate(tag_names):
        if tag == "O":
            flush_active()
            continue

        if "-" not in tag:
            flush_active()
            continue

        prefix, label = tag.split("-", 1)
        if prefix == "B":
            flush_active()
            active_label = label
            active_start_token = i
            active_end_token = i
        elif prefix == "I":
            if active_label == label and active_end_token is not None:
                active_end_token = i
            else:
                flush_active()
                active_label = label
                active_start_token = i
                active_end_token = i
        else:
            flush_active()

    flush_active()
    return entities


def _convert_conll_example(example: Dict[str, object], feature) -> Example:
    tokens = [str(t) for t in example["tokens"]]  # type: ignore[index]
    ner_tags = example["ner_tags"]  # type: ignore[index]
    tag_names = [feature.int2str(int(tag_id)) for tag_id in ner_tags]
    text, offsets = reconstruct_text_and_offsets(tokens)
    gold_entities = bio_to_entities(tokens=tokens, offsets=offsets, tag_names=tag_names)
    return {"text": text, "gold_entities": gold_entities}


def load_conll2003(limit: int = 500, split: str = "validation") -> List[Example]:
    """Load and convert CoNLL-2003 to unified format."""
    load_dataset = _safe_import_datasets()
    dataset = _load_with_compat(load_dataset, "conll2003", split=split)
    dataset = dataset.select(range(min(limit, len(dataset))))
    tag_feature = dataset.features["ner_tags"].feature

    examples: List[Example] = []
    for row in dataset:
        examples.append(_convert_conll_example(row, tag_feature))
    return examples


def try_load_ontonotes(limit: int = 500, split: str = "validation") -> List[Example]:
    """Attempt to load OntoNotes 5.0; return empty list with warning if unavailable."""
    load_dataset = _safe_import_datasets()
    candidate_specs = [
        DatasetSpec(name="conll2012_ontonotesv5", subset="english_v4", split=split),
        DatasetSpec(name="conll2012_ontonotesv5", subset=None, split=split),
    ]

    for spec in candidate_specs:
        try:
            dataset = _load_with_compat(
                load_dataset=load_dataset,
                name=spec.name,
                subset=spec.subset,
                split=spec.split,
            )
            dataset = dataset.select(range(min(limit, len(dataset))))

            if "tokens" not in dataset.features or "named_entities" not in dataset.features:
                warnings.warn(
                    "OntoNotes loaded but schema unsupported for this Phase 1 pipeline."
                )
                return []

            tag_feature = dataset.features["named_entities"].feature
            out: List[Example] = []
            for row in dataset:
                tokens = [str(t) for t in row["tokens"]]
                tags = [tag_feature.int2str(int(tag_id)) for tag_id in row["named_entities"]]
                text, offsets = reconstruct_text_and_offsets(tokens)
                gold = bio_to_entities(tokens=tokens, offsets=offsets, tag_names=tags)
                out.append({"text": text, "gold_entities": gold})
            return out
        except Exception:
            continue

    warnings.warn(
        "OntoNotes 5.0 unavailable in this environment. Skipping OntoNotes benchmark."
    )
    return []
