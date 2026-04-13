"""Evaluation helpers for exact-match NER span scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple


Entity = Dict[str, Any]


def entity_key(entity: Entity) -> Tuple[int, int, str]:
    return (int(entity["start"]), int(entity["end"]), str(entity["label"]))


def precision_recall_f1(gold_entities: List[List[Entity]], pred_entities: List[List[Entity]]) -> Dict[str, float]:
    tp = 0
    fp = 0
    fn = 0

    for gold, pred in zip(gold_entities, pred_entities):
        gold_set: Set[Tuple[int, int, str]] = {entity_key(e) for e in gold}
        pred_set: Set[Tuple[int, int, str]] = {entity_key(e) for e in pred}
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": float(tp), "fp": float(fp), "fn": float(fn)}


def collect_error_examples(
    texts: List[str],
    gold_entities: List[List[Entity]],
    pred_entities: List[List[Entity]],
    max_items: int = 100,
) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    for idx, (text, gold, pred) in enumerate(zip(texts, gold_entities, pred_entities)):
        gset = {entity_key(e) for e in gold}
        pset = {entity_key(e) for e in pred}
        if gset != pset:
            errors.append(
                {
                    "index": idx,
                    "text": text,
                    "gold_entities": gold,
                    "pred_entities": pred,
                    "missing": [list(t) for t in sorted(gset - pset)],
                    "spurious": [list(t) for t in sorted(pset - gset)],
                }
            )
        if len(errors) >= max_items:
            break
    return errors
