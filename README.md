# Phase 1 NER Benchmark Pipeline

Benchmark lightweight NLP models vs LLMs for Named Entity Recognition (NER) on:
- quality (Precision / Recall / F1),
- speed (runtime, avg time, examples/sec),
- and optional cost (token-based when available).

This project compares:
- `spacy`
- `gliner`
- `llama-3.1-8b` (via Ollama)
- `qwen-2.5-7b` (via Ollama)

## Project Files

- `prepare_data.py` - dataset loading + BIO-to-span conversion
- `models.py` - model runners and label normalization
- `evaluate.py` - exact span matching evaluation
- `run_phase1.py` - main pipeline entry point

## 1) Setup

From the `phase1` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "datasets==2.19.2" pandas "spacy<3.8" "gliner<0.3" openai tqdm requests
python -m spacy download en_core_web_sm
```

## 2) LLM Backend Setup (Ollama)

Install and start Ollama:

```bash
brew install ollama
ollama serve
```

In another terminal, pull models:

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

## 3) Run the Benchmark

From `phase1`:

```bash
source .venv/bin/activate
export LLM_BACKEND=ollama
export LLM_MODEL_LLAMA=llama3.1:8b
export LLM_MODEL_QWEN=qwen2.5:7b
python run_phase1.py
```

### Faster test run

```bash
PHASE1_LIMIT=100 python run_phase1.py
```

### Fuller run

```bash
PHASE1_LIMIT=500 python run_phase1.py
```

## 4) Outputs

Generated in `results/`:

- `results.csv` - raw table data
- `results.md` - human-readable comparison table
- `predictions_<model>.json` - per-model predictions
- `error_examples.json` - mismatches for error analysis

## 5) Optional: OpenAI-style API Instead of Ollama

If you want API-based LLMs:

```bash
source .venv/bin/activate
export LLM_BACKEND=openai
export OPENAI_API_KEY="your_key"
export LLM_MODEL_LLAMA="provider_model_id"
export LLM_MODEL_QWEN="provider_model_id"
python run_phase1.py
```

Optional cost settings:

```bash
export LLM_PRICE_PROMPT_PER_1K="0.00015"
export LLM_PRICE_COMPLETION_PER_1K="0.0006"
```

## 6) Notes

- OntoNotes is attempted and skipped gracefully if unavailable/incompatible.
- Labels are normalized to CoNLL-2003 tags: `PER`, `ORG`, `LOC`, `MISC`.
- LLM cost is `null` for local Ollama runs.

## 7) Troubleshooting

- **LLM scores are all `null`:**
  - LLM runner was skipped (check env vars / backend / server).
- **LLM scores are `0.0`:**
  - Model ran but produced poor/invalid extraction for the prompt.
- **Ollama not reachable:**
  - Ensure `ollama serve` is running.
- **CSV hard to read in editor:**
  - Open `results/results.md` instead.

