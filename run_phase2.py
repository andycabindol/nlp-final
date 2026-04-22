"""
Phase 2: Downstream QA Architecture Ablation

Run
---
pip install datasets spacy gliner openai tqdm requests pandas
python -m spacy download en_core_web_sm
python run_phase2.py

LLM backend (Ollama):
  export LLM_BACKEND=ollama
  export LLM_MODEL_LLAMA=llama3.1:8b
  export LLM_MODEL_QWEN=qwen2.5:7b
  ollama pull llama3.1:8b
  ollama pull qwen2.5:7b

Or OpenAI-style API:
  export LLM_BACKEND=openai
  export OPENAI_API_KEY=your_key
  export LLM_MODEL_LLAMA=meta-llama/llama-3.1-8b-instruct
  export LLM_MODEL_QWEN=qwen/qwen-2.5-7b-instruct

Limit for fast testing:
  PHASE2_LIMIT=50 python run_phase2.py
"""

from __future__ import annotations

import json
import math
import os
import re
import string
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Load data

def load_nq(limit: int = 500) -> List[Dict[str, Any]]:
    """Load Natural Questions (single-hop).  Returns list of {question, answer, context}."""
    from datasets import load_dataset  # type: ignore
    print("  Loading NQ validation split...")
    ds = load_dataset("nq_open", split="validation", trust_remote_code=True)
    ds = ds.select(range(min(limit, len(ds))))
    examples = []
    for row in ds:
        # nq_open has "question" and "answer" (list of strings)
        answers = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
        examples.append({
            "question": row["question"],
            "answers": answers,
            "context": row["question"],  # NQ-open has no provided context; question is the query
            "dataset": "nq",
            "hops": 1,
        })
    return examples


def load_hotpotqa(limit: int = 500) -> List[Dict[str, Any]]:
    """Load HotpotQA (two-hop)."""
    from datasets import load_dataset  # type: ignore
    print("  Loading HotpotQA validation split...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    ds = ds.select(range(min(limit, len(ds))))
    examples = []
    for row in ds:
        # Build a flat context from supporting facts
        context_parts = []
        for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
            context_parts.append(f"{title}: " + " ".join(sents))
        context = " ".join(context_parts[:3])  # cap length
        examples.append({
            "question": row["question"],
            "answers": [row["answer"]],
            "context": context,
            "dataset": "hotpotqa",
            "hops": 2,
        })
    return examples


def load_musique(limit: int = 500) -> List[Dict[str, Any]]:
    import json
    print("  Loading MuSiQue from local file...")
    examples = []
    with open("musique_validation.jsonl", "r") as f:
        for line in f:
            row = json.loads(line)
            if not row.get("answerable", True):
                continue
            context = " ".join(p["paragraph_text"] for p in row["paragraphs"][:3])
            examples.append({
                "question": row["question"],
                "answers": [row["answer"]] + row.get("answer_aliases", []),
                "context": context,
                "dataset": "musique",
                "hops": 3,
            })
            if len(examples) >= limit:
                break
    return examples

# NER extractors (mirrors Phase 1)

Entity = Dict[str, Any]


def _normalize_label(raw: str) -> Optional[str]:
    mapping = {
        "PERSON": "PER", "PER": "PER",
        "ORG": "ORG", "ORGANIZATION": "ORG",
        "LOC": "LOC", "GPE": "LOC", "LOCATION": "LOC",
        "MISC": "MISC",
    }
    return mapping.get(raw.strip().upper())


def _normalize_entities(ents: List[Entity], text: str) -> List[Entity]:
    out = []
    for e in ents:
        try:
            s, en, lbl = int(e["start"]), int(e["end"]), str(e["label"])
        except Exception:
            continue
        if not (0 <= s < en <= len(text)):
            continue
        label = _normalize_label(lbl)
        if label is None:
            continue
        out.append({"text": text[s:en], "label": label, "start": s, "end": en})
    return out


class SpacyNER:
    name = "spacy"

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self._nlp = None

    def load(self):
        import spacy  # type: ignore
        self._nlp = spacy.load(self.model_name)

    def extract(self, text: str) -> List[Entity]:
        doc = self._nlp(text)
        raw = [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char} for e in doc.ents]
        return _normalize_entities(raw, text)


class GLiNERNER:
    name = "gliner"

    def __init__(self, model_name: str = "urchade/gliner_medium-v2.1"):
        self.model_name = model_name
        self._model = None
        self._labels = ["person", "organization", "location", "miscellaneous"]

    def load(self):
        from gliner import GLiNER  # type: ignore
        self._model = GLiNER.from_pretrained(self.model_name)

    def extract(self, text: str) -> List[Entity]:
        raw = self._model.predict_entities(text, self._labels)
        ents = [{"text": r.get("text",""), "label": r.get("label",""), "start": r.get("start",-1), "end": r.get("end",-1)} for r in raw]
        return _normalize_entities(ents, text)


class LLMNER:
    """LLM-based NER via Ollama or OpenAI-style API."""

    def __init__(self, name: str, model: str, backend: str = "ollama",
                 api_key: Optional[str] = None, base_url: Optional[str] = None,
                 ollama_url: str = "http://localhost:11434"):
        self.name = name
        self.model = model
        self.backend = backend
        self.api_key = api_key
        self.base_url = base_url
        self.ollama_url = ollama_url.rstrip("/")
        self._client = None
        self.token_usage = {"prompt": 0, "completion": 0}

    def load(self):
        if self.backend == "openai":
            from openai import OpenAI  # type: ignore
            kw = {"api_key": self.api_key}
            if self.base_url:
                kw["base_url"] = self.base_url
            self._client = OpenAI(**kw)

    def _prompt(self, text: str) -> str:
        return (
            "Extract named entities from the text.\n"
            "Return a JSON list with fields: text, label, start, end.\n"
            "Use only labels: PER, ORG, LOC, MISC.\n"
            "No extra text.\n\nText:\n" + text
        )

    def _parse(self, raw: str, text: str) -> List[Entity]:
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1]).strip()
        def _extract(obj):
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                for k in ("entities", "ner", "items", "results"):
                    if isinstance(obj.get(k), list):
                        return obj[k]
            return []
        try:
            return _extract(json.loads(raw))
        except Exception:
            m = re.search(r"\[[\s\S]*\]", raw)
            if m:
                try:
                    return _extract(json.loads(m.group(0)))
                except Exception:
                    pass
        return []

    def extract(self, text: str) -> List[Entity]:
        prompt = self._prompt(text)
        if self.backend == "ollama":
            import requests  # type: ignore
            try:
                resp = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={"model": self.model, "prompt": prompt, "stream": False,
                          "format": "json", "options": {"temperature": 0}},
                    timeout=120,
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "")
            except Exception:
                return []
        else:
            if self._client is None:
                return []
            try:
                response = self._client.responses.create(model=self.model, input=prompt, temperature=0)
                raw = getattr(response, "output_text", "") or ""
                usage = getattr(response, "usage", None)
                if usage:
                    self.token_usage["prompt"] += int(getattr(usage, "input_tokens", 0) or 0)
                    self.token_usage["completion"] += int(getattr(usage, "output_tokens", 0) or 0)
            except Exception:
                return []

        parsed = self._parse(raw, text)
        coerced = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            et = str(item.get("text", "")).strip()
            lbl = str(item.get("label", "")).strip()
            if not et or not lbl:
                continue
            s, en = item.get("start"), item.get("end")
            try:
                s, en = int(s), int(en)
            except Exception:
                idx = text.find(et)
                if idx < 0:
                    continue
                s, en = idx, idx + len(et)
            coerced.append({"text": et, "label": lbl, "start": s, "end": en})
        return _normalize_entities(coerced, text)

# Lightweight knowledge graph

class KnowledgeGraph:
    """
    Minimal LinearRAG-style graph:
      nodes = named entities
      edges = co-occurrence within the same sentence / context window
    """

    def __init__(self):
        self.nodes: Dict[str, str] = {}          # entity_text -> label
        self.edges: List[Tuple[str, str]] = []   # (entity_a, entity_b)

    def add_from_entities(self, entities: List[Entity]):
        texts = [e["text"] for e in entities]
        for e in entities:
            self.nodes[e["text"]] = e["label"]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                self.edges.append((texts[i], texts[j]))

    def retrieve(self, question: str, top_k: int = 10) -> List[str]:
        """Return entity texts that appear in the question (simple keyword match)."""
        q_lower = question.lower()
        hits = [n for n in self.nodes if n.lower() in q_lower]
        # Also return neighbors of matched nodes
        neighbors = set()
        for a, b in self.edges:
            if a in hits:
                neighbors.add(b)
            if b in hits:
                neighbors.add(a)
        combined = list(set(hits) | neighbors)
        return combined[:top_k]


# LinearRAG

class LinearRAGSystem:
    """
    Simplified LinearRAG:
      1. Build graph from context using the chosen NER extractor
      2. Retrieve relevant entities for each question
      3. Generate answer using a lightweight generative LLM
         (same for all three systems — only the graph builder differs)
    """

    def __init__(self, system_name: str, ner_extractor, generation_model: str,
                 generation_backend: str = "ollama",
                 generation_api_key: Optional[str] = None,
                 generation_base_url: Optional[str] = None,
                 generation_ollama_url: str = "http://localhost:11434"):
        self.system_name = system_name
        self.ner = ner_extractor
        self.gen_model = generation_model
        self.gen_backend = generation_backend
        self.gen_api_key = generation_api_key
        self.gen_base_url = generation_base_url
        self.gen_ollama_url = generation_ollama_url.rstrip("/")
        self._gen_client = None
        self.gen_token_usage = {"prompt": 0, "completion": 0}

    def load(self):
        self.ner.load()
        if self.gen_backend == "openai":
            from openai import OpenAI  # type: ignore
            kw = {"api_key": self.gen_api_key}
            if self.gen_base_url:
                kw["base_url"] = self.gen_base_url
            self._gen_client = OpenAI(**kw)

    # Graph construction

    def build_graph(self, context: str) -> Tuple[KnowledgeGraph, float]:
        """Returns (graph, construction_time_sec)."""
        t0 = time.perf_counter()
        graph = KnowledgeGraph()
        # Split context into sentences for more granular co-occurrence
        sentences = re.split(r"(?<=[.!?])\s+", context)
        for sent in sentences:
            entities = self.ner.extract(sent)
            graph.add_from_entities(entities)
        elapsed = time.perf_counter() - t0
        return graph, elapsed

    # Generation

    def _gen_prompt(self, question: str, retrieved_entities: List[str]) -> str:
        entity_str = ", ".join(retrieved_entities) if retrieved_entities else "none found"
        return (
            f"You are a helpful QA assistant.\n"
            f"Relevant entities from the knowledge graph: {entity_str}\n\n"
            f"Question: {question}\n"
            f"Answer concisely in a few words:"
        )

    def generate_answer(self, question: str, retrieved_entities: List[str]) -> str:
        prompt = self._gen_prompt(question, retrieved_entities)
        if self.gen_backend == "ollama":
            import requests  # type: ignore
            try:
                resp = requests.post(
                    f"{self.gen_ollama_url}/api/generate",
                    json={"model": self.gen_model, "prompt": prompt, "stream": False,
                          "options": {"temperature": 0}},
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.json().get("response", "").strip()
            except Exception:
                return ""
        else:
            if self._gen_client is None:
                return ""
            try:
                response = self._gen_client.responses.create(
                    model=self.gen_model, input=prompt, temperature=0
                )
                raw = getattr(response, "output_text", "") or ""
                usage = getattr(response, "usage", None)
                if usage:
                    self.gen_token_usage["prompt"] += int(getattr(usage, "input_tokens", 0) or 0)
                    self.gen_token_usage["completion"] += int(getattr(usage, "output_tokens", 0) or 0)
                return raw.strip()
            except Exception:
                return ""

    # Pipeline

    def run(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run the full pipeline on a list of examples.
        Returns dict with predictions, latency stats, and QA metrics.
        """
        predictions = []
        graph_times = []
        total_nodes = []
        total_edges = []

        for ex in examples:
            graph, g_time = self.build_graph(ex["context"])
            graph_times.append(g_time)
            total_nodes.append(len(graph.nodes))
            total_edges.append(len(graph.edges))

            retrieved = graph.retrieve(ex["question"])
            answer = self.generate_answer(ex["question"], retrieved)

            predictions.append({
                "question": ex["question"],
                "predicted": answer,
                "gold_answers": ex["answers"],
                "dataset": ex["dataset"],
                "hops": ex["hops"],
                "graph_nodes": len(graph.nodes),
                "graph_edges": len(graph.edges),
                "graph_time_sec": g_time,
                "retrieved_entities": retrieved,
            })

        # QA metrics
        em_scores = [exact_match(p["predicted"], p["gold_answers"]) for p in predictions]
        f1_scores = [token_f1(p["predicted"], p["gold_answers"]) for p in predictions]

        n = max(len(predictions), 1)
        total_graph_time = sum(graph_times)

        # NER token cost (only populated for LLMNER extractors)
        ner_prompt_tokens = getattr(self.ner, "token_usage", {}).get("prompt", 0)
        ner_completion_tokens = getattr(self.ner, "token_usage", {}).get("completion", 0)
        total_prompt_tokens = self.gen_token_usage["prompt"] + ner_prompt_tokens
        total_completion_tokens = self.gen_token_usage["completion"] + ner_completion_tokens

        return {
            "system": self.system_name,
            "n_examples": len(predictions),
            "em": sum(em_scores) / n,
            "f1": sum(f1_scores) / n,
            "avg_graph_time_sec": total_graph_time / n,
            "total_graph_time_sec": total_graph_time,
            "examples_per_sec": len(predictions) / total_graph_time if total_graph_time > 0 else 0.0,
            "avg_graph_nodes": sum(total_nodes) / n,
            "avg_graph_edges": sum(total_edges) / n,
            "ner_prompt_tokens": ner_prompt_tokens,
            "ner_completion_tokens": ner_completion_tokens,
            "gen_prompt_tokens": self.gen_token_usage["prompt"],
            "gen_completion_tokens": self.gen_token_usage["completion"],
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "predictions": predictions,
        }

# QA evaluation

def _normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation/articles (standard SQuAD normalization)."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def exact_match(prediction: str, gold_answers: List[str]) -> float:
    pred_norm = _normalize_answer(prediction)
    return float(any(_normalize_answer(g) == pred_norm for g in gold_answers))


def token_f1(prediction: str, gold_answers: List[str]) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = _normalize_answer(gold).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens) if pred_tokens else 0.0
        recall = num_same / len(gold_tokens) if gold_tokens else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        best_f1 = max(best_f1, f1)
    return best_f1

# Per-dataset breakdown

def breakdown_by_dataset(predictions: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List] = {}
    for p in predictions:
        buckets.setdefault(p["dataset"], []).append(p)
    out = {}
    for ds, preds in buckets.items():
        ems = [exact_match(p["predicted"], p["gold_answers"]) for p in preds]
        f1s = [token_f1(p["predicted"], p["gold_answers"]) for p in preds]
        n = max(len(preds), 1)
        out[ds] = {
            "n": len(preds),
            "em": sum(ems) / n,
            "f1": sum(f1s) / n,
        }
    return out


# Output

def fmt(v: Any, decimals: int = 4) -> str:
    if v is None:
        return "null"
    try:
        fv = float(v)
    except Exception:
        return str(v)
    if math.isnan(fv):
        return "null"
    return f"{fv:.{decimals}f}"


def save_results(rows: List[Dict[str, Any]], results_dir: Path):
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(rows)
        df.to_csv(results_dir / "phase2_results.csv", index=False)
    except Exception as exc:
        warnings.warn(f"pandas not available, skipping CSV: {exc}")


def print_summary_table(rows: List[Dict[str, Any]]):
    headers = ["System", "EM", "F1", "Avg Graph Time (s)", "Examples/sec", "Avg Nodes", "Avg Edges", "Total Tokens"]
    col_w = [max(len(h), 18) for h in headers]

    def row_str(cells):
        return " | ".join(str(c).ljust(col_w[i]) for i, c in enumerate(cells))

    print("\n" + "=" * 140)
    print("Phase 2 Results Summary")
    print("=" * 140)
    print(row_str(headers))
    print("-+-".join("-" * w for w in col_w))
    for r in rows:
        total_tokens = r.get("total_prompt_tokens", 0) + r.get("total_completion_tokens", 0)
        print(row_str([
            r["system"],
            fmt(r["em"]),
            fmt(r["f1"]),
            fmt(r["avg_graph_time_sec"]),
            fmt(r["examples_per_sec"]),
            fmt(r["avg_graph_nodes"]),
            fmt(r["avg_graph_edges"]),
            str(int(total_tokens)) if total_tokens else "N/A",
        ]))
    print("=" * 140)


def save_markdown(rows: List[Dict[str, Any]], results_dir: Path):
    lines = [
        "# Phase 2 Results\n",
        "## Overall QA + Graph Construction",
        "| System | EM | F1 | Avg Graph Time (s) | Examples/sec | Avg Nodes | Avg Edges | NER Tokens | Gen Tokens | Total Tokens |",
        "|--------|----|----|--------------------|--------------|-----------|-----------|------------|------------|--------------|",
    ]
    for r in rows:
        ner_tokens = r.get("ner_prompt_tokens", 0) + r.get("ner_completion_tokens", 0)
        gen_tokens = r.get("gen_prompt_tokens", 0) + r.get("gen_completion_tokens", 0)
        total_tokens = r.get("total_prompt_tokens", 0) + r.get("total_completion_tokens", 0)
        lines.append(
            f"| {r['system']} | {fmt(r['em'])} | {fmt(r['f1'])} | "
            f"{fmt(r['avg_graph_time_sec'])} | {fmt(r['examples_per_sec'])} | "
            f"{fmt(r['avg_graph_nodes'])} | {fmt(r['avg_graph_edges'])} | "
            f"{int(ner_tokens) if ner_tokens else 'N/A'} | "
            f"{int(gen_tokens) if gen_tokens else 'N/A'} | "
            f"{int(total_tokens) if total_tokens else 'N/A'} |"
        )
    lines.append("")
    (results_dir / "phase2_results.md").write_text("\n".join(lines), encoding="utf-8")

# Main

def main():
    limit = int(os.getenv("PHASE2_LIMIT", "500"))
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    all_examples = []
    all_examples += load_nq(limit=limit)
    all_examples += load_hotpotqa(limit=limit)
    all_examples += load_musique(limit=limit)
    print(f"  Total examples: {len(all_examples)}")

    # LLM config
    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llama_model = os.getenv("LLM_MODEL_LLAMA", "llama3.1:8b")
    qwen_model = os.getenv("LLM_MODEL_QWEN", "qwen2.5:7b")
    # Generation model is shared across all three systems (same downstream LLM)
    gen_model = os.getenv("GEN_MODEL", llama_model)

    # Define the three systems
    systems = [
        LinearRAGSystem(
            system_name="LinearRAG_src (SpaCy)",
            ner_extractor=SpacyNER(),
            generation_model=gen_model,
            generation_backend=backend,
            generation_api_key=api_key,
            generation_base_url=base_url,
            generation_ollama_url=ollama_url,
        ),
        LinearRAGSystem(
            system_name="LinearRAG_NLP (GLiNER)",
            ner_extractor=GLiNERNER(),
            generation_model=gen_model,
            generation_backend=backend,
            generation_api_key=api_key,
            generation_base_url=base_url,
            generation_ollama_url=ollama_url,
        ),
        LinearRAGSystem(
            system_name="LinearRAG_LLM (Llama)",
            ner_extractor=LLMNER(
                name="llama-ner", model=llama_model, backend=backend,
                api_key=api_key, base_url=base_url, ollama_url=ollama_url,
            ),
            generation_model=gen_model,
            generation_backend=backend,
            generation_api_key=api_key,
            generation_base_url=base_url,
            generation_ollama_url=ollama_url,
        ),
        LinearRAGSystem(
            system_name="LinearRAG_LLM (Qwen)",
            ner_extractor=LLMNER(
                name="qwen-ner", model=qwen_model, backend=backend,
                api_key=api_key, base_url=base_url, ollama_url=ollama_url,
            ),
            generation_model=gen_model,
            generation_backend=backend,
            generation_api_key=api_key,
            generation_base_url=base_url,
            generation_ollama_url=ollama_url,
        ),
    ]

    # Run each system
    all_rows = []
    for system in systems:
        print(f"\nRunning {system.system_name}...")
        try:
            system.load()
        except Exception as exc:
            warnings.warn(f"  Could not load {system.system_name}: {exc}. Skipping.")
            continue

        result = system.run(all_examples)

        # Per-dataset breakdown
        breakdown = breakdown_by_dataset(result["predictions"])
        result["breakdown"] = breakdown

        # Save predictions
        pred_path = results_dir / f"predictions_{system.system_name.replace(' ', '_').replace('(', '').replace(')', '')}.json"
        with pred_path.open("w", encoding="utf-8") as f:
            json.dump(result["predictions"], f, ensure_ascii=False, indent=2)

        # Print per-dataset breakdown
        print(f"  Per-dataset results:")
        for ds, stats in breakdown.items():
            print(f"    {ds}: n={stats['n']}  EM={fmt(stats['em'])}  F1={fmt(stats['f1'])}")

        all_rows.append(result)

    # Print summary
    if not all_rows:
        print("\nNo systems produced results. Check your environment setup.")
        return

    summary_rows = [{k: v for k, v in r.items() if k != "predictions"} for r in all_rows]
    save_results(summary_rows, results_dir)
    save_markdown(summary_rows, results_dir)
    print_summary_table(summary_rows)

    # Combined error examples
    error_examples = []
    for result in all_rows:
        for p in result["predictions"]:
            em = exact_match(p["predicted"], p["gold_answers"])
            if em == 0.0:
                error_examples.append({
                    "system": result["system"],
                    "question": p["question"],
                    "predicted": p["predicted"],
                    "gold_answers": p["gold_answers"],
                    "dataset": p["dataset"],
                    "hops": p["hops"],
                    "retrieved_entities": p["retrieved_entities"],
                })
            if len(error_examples) >= 300:
                break

    with (results_dir / "error_examples.json").open("w", encoding="utf-8") as f:
        json.dump(error_examples, f, ensure_ascii=False, indent=2)

    print(f"\nSaved all artifacts to: {results_dir.resolve()}")
    print(f"  - phase2_results.csv")
    print(f"  - phase2_results.md")
    print(f"  - predictions_<system>.json")
    print(f"  - error_examples.json")


if __name__ == "__main__":
    main()
