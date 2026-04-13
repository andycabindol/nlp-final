"""
Phase 1 NER Benchmarking Pipeline
=================================

Dependencies
------------
- Python 3.9+
- datasets
- pandas
- spacy
- gliner
- openai (optional, for LLM baseline)
- tqdm

Install
-------
pip install datasets pandas spacy gliner openai tqdm
python -m spacy download en_core_web_sm

Run
---
python run_phase1.py

Enable LLM Baseline
-------------------
Set environment variables before running:

export OPENAI_API_KEY="your_api_key"
export LLM_MODEL="gpt-4.1-mini"

For local Ollama comparisons (proposal models):
export LLM_BACKEND="ollama"
export LLM_MODEL_LLAMA="llama3.1:8b"
export LLM_MODEL_QWEN="qwen2.5:7b"

Optional pricing for cost estimates:
export LLM_PRICE_PROMPT_PER_1K="0.00015"
export LLM_PRICE_COMPLETION_PER_1K="0.0006"
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List
import warnings

from evaluate import collect_error_examples, precision_recall_f1
from models import GLiNERRunner, LLMRunner, SpacyRunner
from prepare_data import load_conll2003, try_load_ontonotes


def ensure_results_dir() -> Path:
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _to_float_or_none(value: Any) -> Any:
    try:
        fv = float(value)
    except Exception:
        return None
    if math.isnan(fv):
        return None
    return fv


def run_model_on_dataset(
    model_runner,
    dataset_name: str,
    examples: List[Dict[str, Any]],
    results_dir: Path,
) -> Dict[str, Any]:
    available, reason = model_runner.available()
    if not available:
        warnings.warn(f"[{model_runner.name}] skipped on {dataset_name}: {reason}")
        return {
            "model": model_runner.name,
            "dataset": dataset_name,
            "precision": None,
            "recall": None,
            "f1": None,
            "avg_time": None,
            "examples_per_sec": None,
            "cost": None,
            "skipped_reason": reason,
        }

    texts = [item["text"] for item in examples]
    gold = [item["gold_entities"] for item in examples]

    preds, speed = model_runner.run_batch(texts)
    prf = precision_recall_f1(gold_entities=gold, pred_entities=preds)

    model_pred_payload = []
    for i, (text, g, p) in enumerate(zip(texts, gold, preds)):
        model_pred_payload.append(
            {"index": i, "text": text, "gold_entities": g, "pred_entities": p}
        )

    pred_path = results_dir / f"predictions_{model_runner.name}_{dataset_name}.json"
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(model_pred_payload, f, ensure_ascii=False, indent=2)

    errors = collect_error_examples(texts=texts, gold_entities=gold, pred_entities=preds)
    error_path = results_dir / f"error_examples_{model_runner.name}_{dataset_name}.json"
    with error_path.open("w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    return {
        "model": model_runner.name,
        "dataset": dataset_name,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "avg_time": speed.get("avg_runtime_sec"),
        "examples_per_sec": speed.get("examples_per_sec"),
        "cost": _to_float_or_none(speed.get("cost_usd")),
        "prompt_tokens": _to_float_or_none(speed.get("prompt_tokens")),
        "completion_tokens": _to_float_or_none(speed.get("completion_tokens")),
    }


def save_results_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas is required for saving results.csv") from exc

    df = pd.DataFrame(rows)
    preferred_order = [
        "model",
        "dataset",
        "precision",
        "recall",
        "f1",
        "avg_time",
        "examples_per_sec",
        "cost",
        "prompt_tokens",
        "completion_tokens",
        "skipped_reason",
    ]
    ordered_cols = [c for c in preferred_order if c in df.columns] + [
        c for c in df.columns if c not in preferred_order
    ]
    df = df[ordered_cols]
    df.to_csv(output_path, index=False)


def save_results_markdown(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Save a more legible markdown table for quick viewing in editors."""
    headers = [
        "Model",
        "Dataset",
        "Precision",
        "Recall",
        "F1",
        "Avg Time",
        "Examples/sec",
        "Cost",
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append(
            "| {model} | {dataset} | {precision} | {recall} | {f1} | {avg_time} | {examples_per_sec} | {cost} |".format(
                model=row.get("model"),
                dataset=row.get("dataset"),
                precision=fmt_float(row.get("precision")),
                recall=fmt_float(row.get("recall")),
                f1=fmt_float(row.get("f1")),
                avg_time=fmt_float(row.get("avg_time")),
                examples_per_sec=fmt_float(row.get("examples_per_sec")),
                cost=fmt_float(row.get("cost")),
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_legible_table(rows: List[Dict[str, Any]]) -> None:
    """Print an aligned plain-text table for terminal readability."""
    headers = ["Model", "Dataset", "Precision", "Recall", "F1", "Avg Time", "Examples/sec", "Cost"]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                str(row.get("model")),
                str(row.get("dataset")),
                fmt_float(row.get("precision")),
                fmt_float(row.get("recall")),
                fmt_float(row.get("f1")),
                fmt_float(row.get("avg_time")),
                fmt_float(row.get("examples_per_sec")),
                fmt_float(row.get("cost")),
            ]
        )

    widths = [len(h) for h in headers]
    for r in table_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def _fmt_row(cells: List[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print("\nFinal Comparison Table (Legible)")
    print(_fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for r in table_rows:
        print(_fmt_row(r))


def print_comparison_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "Model",
        "Dataset",
        "Precision",
        "Recall",
        "F1",
        "Avg Time",
        "Examples/sec",
        "Cost",
    ]
    print("\nFinal Comparison Table")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print(
            "| {model} | {dataset} | {precision} | {recall} | {f1} | {avg_time} | {examples_per_sec} | {cost} |".format(
                model=row.get("model"),
                dataset=row.get("dataset"),
                precision=fmt_float(row.get("precision")),
                recall=fmt_float(row.get("recall")),
                f1=fmt_float(row.get("f1")),
                avg_time=fmt_float(row.get("avg_time")),
                examples_per_sec=fmt_float(row.get("examples_per_sec")),
                cost=fmt_float(row.get("cost")),
            )
        )


def fmt_float(v: Any) -> str:
    if v is None:
        return "null"
    try:
        fv = float(v)
    except Exception:
        return str(v)
    if math.isnan(fv):
        return "null"
    return f"{fv:.4f}"


def main() -> None:
    limit = int(os.getenv("PHASE1_LIMIT", "500"))
    split = os.getenv("PHASE1_SPLIT", "validation")

    print(f"Loading CoNLL-2003 ({split}, limit={limit})...")
    conll_examples = load_conll2003(limit=limit, split=split)

    print(f"Trying OntoNotes 5.0 ({split}, limit={limit})...")
    ontonotes_examples = try_load_ontonotes(limit=limit, split=split)

    datasets = [("conll2003", conll_examples)]
    if ontonotes_examples:
        datasets.append(("ontonotes5", ontonotes_examples))

    pricing_prompt = _to_float_or_none(os.getenv("LLM_PRICE_PROMPT_PER_1K"))
    pricing_completion = _to_float_or_none(os.getenv("LLM_PRICE_COMPLETION_PER_1K"))
    llama_model_name = os.getenv("LLM_MODEL_LLAMA", "llama3.1:8b")
    qwen_model_name = os.getenv("LLM_MODEL_QWEN", "qwen2.5:7b")

    runners = [
        SpacyRunner(),
        GLiNERRunner(),
        LLMRunner(
            runner_name="llama-3.1-8b",
            model=llama_model_name,
            pricing_per_1k_prompt=pricing_prompt,
            pricing_per_1k_completion=pricing_completion,
        ),
        LLMRunner(
            runner_name="qwen-2.5-7b",
            model=qwen_model_name,
            pricing_per_1k_prompt=pricing_prompt,
            pricing_per_1k_completion=pricing_completion,
        ),
    ]

    results_dir = ensure_results_dir()
    final_rows: List[Dict[str, Any]] = []
    all_errors: List[Dict[str, Any]] = []
    aggregated_predictions: Dict[str, List[Dict[str, Any]]] = {}

    for dataset_name, examples in datasets:
        print(f"\nBenchmarking on {dataset_name}: {len(examples)} examples")
        for runner in runners:
            print(f"- Running {runner.name}...")
            row = run_model_on_dataset(
                model_runner=runner,
                dataset_name=dataset_name,
                examples=examples,
                results_dir=results_dir,
            )
            final_rows.append(row)

            pred_file = results_dir / f"predictions_{runner.name}_{dataset_name}.json"
            if pred_file.exists():
                with pred_file.open("r", encoding="utf-8") as f:
                    items = json.load(f)
                for item in items:
                    item["dataset"] = dataset_name
                aggregated_predictions.setdefault(runner.name, []).extend(items)

            err_file = results_dir / f"error_examples_{runner.name}_{dataset_name}.json"
            if err_file.exists():
                with err_file.open("r", encoding="utf-8") as f:
                    all_errors.extend(json.load(f))

    save_results_csv(final_rows, results_dir / "results.csv")
    for model_name, payload in aggregated_predictions.items():
        with (results_dir / f"predictions_{model_name}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    with (results_dir / "error_examples.json").open("w", encoding="utf-8") as f:
        json.dump(all_errors, f, ensure_ascii=False, indent=2)

    print_comparison_table(final_rows)
    print_legible_table(final_rows)
    save_results_markdown(final_rows, results_dir / "results.md")
    print(f"\nSaved artifacts to: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
