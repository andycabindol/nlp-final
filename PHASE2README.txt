After running phase 1 and compiling properly. Download the dataset musique from https://huggingface.co/datasets/dgslibisey/MuSiQue, rename it to "musique_validation.jsonl". Musique can't be stored in GitHub repository due to GitHub size restraints.

Following Phase 1, we use 
source .venv/bin/activate
export LLM_BACKEND=ollama
export LLM_MODEL_LLAMA=llama3.1:8b
export LLM_MODEL_QWEN=qwen2.5:7b
But instead of running Phase 1 again we instead use "PHASE2_LIMIT=500 python run_phase2.py" to run Phase 2. 

Output will be phase2_results.csv, phase2_results.md, predictions_LinearRAG_LLM_Llama.json, predictions_LinearRAG_LLM_Qwen.json, predictions_LinearRAG_NLP_GLiNER.json, predictions_LinearRAG_src_SpaCy.json, error_examples.json.