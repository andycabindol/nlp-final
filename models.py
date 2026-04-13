"""Model runners for spaCy, GLiNER, and LLM baselines."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple


Entity = Dict[str, Any]


def normalize_label(raw_label: str) -> Optional[str]:
    """Map model-specific labels to CoNLL-2003 labels."""
    label = raw_label.strip().upper()
    mapping = {
        "PERSON": "PER",
        "PER": "PER",
        "ORG": "ORG",
        "ORGANIZATION": "ORG",
        "LOC": "LOC",
        "GPE": "LOC",
        "LOCATION": "LOC",
        "MISC": "MISC",
    }
    return mapping.get(label)


def normalize_entities(entities: List[Entity], text: str) -> List[Entity]:
    out: List[Entity] = []
    for ent in entities:
        try:
            start = int(ent["start"])
            end = int(ent["end"])
            raw_label = str(ent["label"])
        except Exception:
            continue
        if not (0 <= start < end <= len(text)):
            continue
        label = normalize_label(raw_label)
        if label is None:
            continue
        out.append(
            {
                "text": text[start:end],
                "label": label,
                "start": start,
                "end": end,
            }
        )
    return out


def _find_offsets_for_entity_text(text: str, entity_text: str) -> Optional[Tuple[int, int]]:
    """Find best-effort char offsets when model omits start/end."""
    candidate = (entity_text or "").strip()
    if not candidate:
        return None
    start = text.find(candidate)
    if start >= 0:
        return start, start + len(candidate)
    # Case-insensitive fallback for models that change casing.
    match = re.search(re.escape(candidate), text, flags=re.IGNORECASE)
    if match:
        return match.start(), match.end()
    return None


def coerce_entities_to_schema(raw_entities: Any, text: str) -> List[Entity]:
    """Accept loose LLM outputs and coerce into {text,label,start,end}."""
    if not isinstance(raw_entities, list):
        return []
    coerced: List[Entity] = []
    for item in raw_entities:
        if not isinstance(item, dict):
            continue

        ent_text = str(item.get("text", "") or "").strip()
        label = str(item.get("label", "") or "").strip()
        if not ent_text or not label:
            continue

        start = item.get("start")
        end = item.get("end")
        try:
            start_i = int(start) if start is not None else None
            end_i = int(end) if end is not None else None
        except Exception:
            start_i = None
            end_i = None

        # If offsets are missing or invalid, infer them from entity text.
        if (
            start_i is None
            or end_i is None
            or start_i < 0
            or end_i <= start_i
            or end_i > len(text)
        ):
            inferred = _find_offsets_for_entity_text(text, ent_text)
            if inferred is None:
                continue
            start_i, end_i = inferred

        coerced.append(
            {
                "text": ent_text,
                "label": label,
                "start": start_i,
                "end": end_i,
            }
        )
    return coerced


class BaseRunner:
    name: str = "base"

    def available(self) -> Tuple[bool, str]:
        return True, ""

    def predict(self, text: str) -> List[Entity]:
        raise NotImplementedError

    def run_batch(self, texts: List[str]) -> Tuple[List[List[Entity]], Dict[str, float]]:
        started = time.perf_counter()
        predictions: List[List[Entity]] = []
        for text in texts:
            predictions.append(self.predict(text))
        total_time = time.perf_counter() - started
        n = max(len(texts), 1)
        metrics = {
            "total_runtime_sec": total_time,
            "avg_runtime_sec": total_time / n,
            "examples_per_sec": len(texts) / total_time if total_time > 0 else 0.0,
        }
        return predictions, metrics


class SpacyRunner(BaseRunner):
    name = "spacy"

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self.nlp = None

    def available(self) -> Tuple[bool, str]:
        try:
            import spacy  # type: ignore

            self.nlp = spacy.load(self.model_name)
            return True, ""
        except Exception as exc:
            return False, f"spaCy unavailable: {exc}"

    def predict(self, text: str) -> List[Entity]:
        assert self.nlp is not None
        doc = self.nlp(text)
        entities = [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]
        return normalize_entities(entities, text)


class GLiNERRunner(BaseRunner):
    name = "gliner"

    def __init__(self, model_name: str = "urchade/gliner_medium-v2.1") -> None:
        self.model_name = model_name
        self.model = None
        self.target_labels = ["person", "organization", "location", "miscellaneous"]

    def available(self) -> Tuple[bool, str]:
        try:
            from gliner import GLiNER  # type: ignore

            self.model = GLiNER.from_pretrained(self.model_name)
            return True, ""
        except Exception as exc:
            return False, f"GLiNER unavailable: {exc}"

    def predict(self, text: str) -> List[Entity]:
        assert self.model is not None
        raw = self.model.predict_entities(text, self.target_labels)
        entities: List[Entity] = []
        for item in raw:
            entities.append(
                {
                    "text": item.get("text", ""),
                    "label": item.get("label", ""),
                    "start": item.get("start", -1),
                    "end": item.get("end", -1),
                }
            )
        return normalize_entities(entities, text)


class LLMRunner(BaseRunner):
    """Swappable LLM NER runner using OpenAI-style client or local stub."""

    name = "llm"

    def __init__(
        self,
        runner_name: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        pricing_per_1k_prompt: Optional[float] = None,
        pricing_per_1k_completion: Optional[float] = None,
    ) -> None:
        if runner_name:
            self.name = runner_name
        self.model = model or os.getenv("LLM_MODEL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.backend = os.getenv("LLM_BACKEND", "openai").strip().lower()
        self.openai_base_url = os.getenv("OPENAI_BASE_URL")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.client = None
        self.pricing_prompt = pricing_per_1k_prompt
        self.pricing_completion = pricing_per_1k_completion
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def available(self) -> Tuple[bool, str]:
        if not self.model:
            return False, "LLM skipped: missing model (set LLM_MODEL)."
        if self.backend == "ollama":
            try:
                import requests  # type: ignore

                # Lightweight health check so we fail clearly if ollama isn't running.
                requests.get(f"{self.ollama_base_url}/api/tags", timeout=3)
                return True, ""
            except Exception as exc:
                return False, f"LLM skipped: Ollama unavailable ({exc})."

        if not self.api_key:
            return False, "LLM skipped: missing API key (set OPENAI_API_KEY)."
        try:
            from openai import OpenAI  # type: ignore

            client_kwargs = {"api_key": self.api_key}
            if self.openai_base_url:
                client_kwargs["base_url"] = self.openai_base_url
            self.client = OpenAI(**client_kwargs)
            return True, ""
        except Exception as exc:
            return False, f"LLM skipped: OpenAI client unavailable ({exc})."

    def _build_prompt(self, text: str) -> str:
        return (
            "Extract named entities from the text.\n"
            "Return JSON list with fields: text, label, start, end.\n"
            "Use labels from {PER, ORG, LOC, MISC} when possible.\n"
            "Do not include any extra text.\n\n"
            f"Text:\n{text}"
        )

    def _local_stub(self, text: str) -> List[Entity]:
        # Simple fallback if API call fails unexpectedly.
        return []

    def _parse_response_json(self, raw_text: str) -> List[Entity]:
        candidate = (raw_text or "").strip()
        if not candidate:
            return []

        # Strip markdown fences when models wrap JSON in ```json ... ```
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if len(lines) >= 3:
                candidate = "\n".join(lines[1:-1]).strip()

        def _extract_entities(parsed_obj: Any) -> List[Entity]:
            # Accept either direct list, or wrapped payloads like {"entities":[...]}.
            if isinstance(parsed_obj, list):
                return parsed_obj
            if isinstance(parsed_obj, dict):
                # Some models emit a single entity object directly.
                if "text" in parsed_obj and "label" in parsed_obj:
                    return [parsed_obj]
                for key in ("entities", "ner", "items", "results"):
                    value = parsed_obj.get(key)
                    if isinstance(value, list):
                        return value
            return []

        try:
            parsed = json.loads(candidate)
            return _extract_entities(parsed)
        except Exception:
            # Best-effort salvage: find first JSON array/object in free text.
            array_match = re.search(r"\[[\s\S]*\]", candidate)
            if array_match:
                try:
                    parsed = json.loads(array_match.group(0))
                    return _extract_entities(parsed)
                except Exception:
                    pass
            obj_match = re.search(r"\{[\s\S]*\}", candidate)
            if obj_match:
                try:
                    parsed = json.loads(obj_match.group(0))
                    return _extract_entities(parsed)
                except Exception:
                    pass
            return []

    def _predict_via_ollama(self, text: str, prompt: str) -> List[Entity]:
        try:
            import requests  # type: ignore
        except Exception:
            return []

        try:
            resp = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0},
                },
                timeout=120,
            )
            resp.raise_for_status()
            payload = resp.json()
            raw_text = payload.get("response", "")
            parsed = self._parse_response_json(raw_text)
            coerced = coerce_entities_to_schema(parsed, text)
            return normalize_entities(coerced, text)
        except Exception:
            return []

    def predict(self, text: str) -> List[Entity]:
        prompt = self._build_prompt(text)
        if self.backend == "ollama":
            return self._predict_via_ollama(text, prompt)

        if self.client is None:
            return []

        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                temperature=0,
            )
            raw_text = getattr(response, "output_text", "") or ""
            usage = getattr(response, "usage", None)
            if usage:
                self.token_usage["prompt_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
                self.token_usage["completion_tokens"] += int(
                    getattr(usage, "output_tokens", 0) or 0
                )

            parsed = self._parse_response_json(raw_text)
            coerced = coerce_entities_to_schema(parsed, text)
            return normalize_entities(coerced, text)
        except Exception:
            return self._local_stub(text)

    def run_batch(self, texts: List[str]) -> Tuple[List[List[Entity]], Dict[str, float]]:
        preds, metrics = super().run_batch(texts)
        prompt_tokens = self.token_usage["prompt_tokens"]
        completion_tokens = self.token_usage["completion_tokens"]
        metrics["prompt_tokens"] = float(prompt_tokens)
        metrics["completion_tokens"] = float(completion_tokens)

        if self.pricing_prompt is not None and self.pricing_completion is not None:
            cost = (prompt_tokens / 1000.0) * self.pricing_prompt + (
                completion_tokens / 1000.0
            ) * self.pricing_completion
            metrics["cost_usd"] = cost
        else:
            metrics["cost_usd"] = float("nan")
        return preds, metrics
