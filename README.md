# ChatGeo-Magi

<a href="https://doi.org/10.5281/zenodo.21535623"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.21535623.svg" alt="DOI"></a>

This repository contains all the code of ChatGeo-Magi, including the agentic approach, two-LLM approach, evaluation, plotting, and the testing website.

ChatGeo-Magi is an AI agent for natural-language access to NOAA/CIRES geomagnetic data, models, and resources. It combines RAG with structured tool calling to answer questions, run API calculations, and generate visualizations.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) for the local models (`gpt-oss:20b`, `llama3.3:70b`, `mistral-nemo:12b`)
- An OpenAI API key for reference-aware evaluation judging (only needed when rerunning LLM judge evaluations)
- Public geomagnetic documentation (PDFs or text files) added to `agentic/data/` to form the knowledge base

## Inferencing

To get the system running, install the dependencies, configure environment variables, and start the application:

```bash
cd agentic

# Set API keys via environment variables or a .env file (do not hardcode keys in Python files):
# export CALC_API_KEY="your_noaa_calc_key"
# export GEOCODE_API_KEY="your_google_maps_key"
#
# Review model and retrieval settings in config.py and populate data/

# It is recommended to create a virtual environment to avoid package conflicts:
# python -m venv venv
# source venv/bin/activate

pip install -r requirements.txt
python cgmterm.py           # for a terminal interface
streamlit run cgmwebsite.py # for a web interface
```

## Reproducing Evaluation

The evaluation directory provides two separate workflows: an offline plotting workflow that regenerates all manuscript figures directly from the published result tables, and a full benchmark runner for rerunning model inference and scoring.

### 1. Regenerate Manuscript Figures (Offline, No API Keys or GPU Required)

Install the lightweight plotting dependencies and run [`evaluation/plots.py`](evaluation/plots.py) from the repository root:

```bash
pip install -r evaluation/requirements.txt
python evaluation/plots.py
```

To regenerate a single figure while editing plot styling:

```bash
python evaluation/plots.py --figure rag
```

Available `--figure` choices are `all` (default), `retrieval`, `rag`, `questions`, `tools`, and `london`.

- **Generated figures** (`.pdf` and `.png`) are written to [`evaluation/plot_outputs/`](evaluation/plot_outputs/).
- **Published result tables and summaries** live in [`evaluation/results/`](evaluation/results/).
- **Golden questions and benchmark definitions** live in [`evaluation/goldens.csv`](evaluation/goldens.csv) and [`evaluation/benchmarks/`](evaluation/benchmarks/).
- **Historical email/no-email experiment archives** from the initial submission are isolated in [`evaluation/legacy_email_experiment/`](evaluation/legacy_email_experiment/).

### 2. Rerun Full Experiments (Requires Ollama/GPU, Source PDFs, and API Keys)

Rerunning the live benchmarks from scratch requires:
- Local [`gpt-oss:20b`](https://ollama.com) via Ollama and the `NovaSearch/stella_en_1.5B_v5` embedding model.
- The 15 public NOAA/BGS source documents listed with their source URLs and SHA-256 hashes in [`evaluation/benchmarks/public_corpus_manifest.json`](evaluation/benchmarks/public_corpus_manifest.json).
- `CALC_API_KEY` (and optional `GEOCODE_API_KEY`) for live NOAA geomagnetic calculator queries, plus `OPENAI_API_KEY` for `gpt-4o-mini` reference judging.

```bash
# 1. Build the normalized cosine Stella Chroma index from the public corpus
python evaluation/build_public_index.py \
  --manifest evaluation/benchmarks/public_corpus_manifest.json \
  --data-dir /path/to/public_pdfs \
  --index-dir /tmp/chatgeomagi_index \
  --embedding-model NovaSearch/stella_en_1.5B_v5 \
  --preset stella_s2p_cosine

# 2. Score retrieval performance (Hit@4, Recall@4, MRR@4)
python evaluation/score_retrieval.py \
  --benchmark evaluation/benchmarks/retrieval.json \
  --manifest /tmp/chatgeomagi_index/indexed_manifest.json \
  --index-dir /tmp/chatgeomagi_index \
  --embedding-model NovaSearch/stella_en_1.5B_v5 \
  --preset stella_s2p_cosine \
  --output /tmp/chatgeomagi_eval/retrieval_scores.json

# 3. Run paired No-RAG vs. RAG inference on the 25 golden questions
python evaluation/run_revision_inference.py \
  --model gpt-oss:20b \
  --manifest /tmp/chatgeomagi_index/indexed_manifest.json \
  --index-dir /tmp/chatgeomagi_index \
  --embedding-model NovaSearch/stella_en_1.5B_v5 \
  --preset stella_s2p_cosine \
  --output-dir /tmp/chatgeomagi_eval/ablation

# 4. Score paired outputs with the reference-aware GPT-4o-mini judge
python evaluation/openai_reference_judge.py \
  --inputs /tmp/chatgeomagi_eval/ablation/judge_inputs.json \
  --output-dir /tmp/chatgeomagi_eval/judging

# 5. Run and score the 20-prompt live NOAA tool-calling benchmark
python evaluation/run_tool_benchmark.py \
  --model gpt-oss:20b \
  --manifest /tmp/chatgeomagi_index/indexed_manifest.json \
  --index-dir /tmp/chatgeomagi_index \
  --embedding-model NovaSearch/stella_en_1.5B_v5 \
  --output /tmp/chatgeomagi_eval/tool_traces.json

python evaluation/score_live_tools.py \
  --traces /tmp/chatgeomagi_eval/tool_traces.json \
  --output-dir /tmp/chatgeomagi_eval/tool_scores
```
