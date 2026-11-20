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
# export GEOCODE_API_KEY="your_geocode_maps_co_key"  # https://geocode.maps.co
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

The email-condition CSV and three corresponding Opik JSONs contain responses generated with restricted support-email retrieval. Inspection found no raw messages, contact details, or credentials, but cannot establish permission to publish derived text. Author/NOAA permission review remains necessary; relocation does not remove earlier public Git copies.

### 2. Rerun Scientific Experiments

These commands collect new results in a fresh temporary directory. They do not update `evaluation/results/` or the manuscript figures. Run them from the repository root in a Python 3.11 environment with `agentic/requirements.txt` and `evaluation/requirements.txt` installed. Retrieval also requires the cached local Stella directory; inference requires a running Ollama server and sufficient GPU memory (the experiments used a 40 GB A100).

Required setup:

- Install the exact `gpt-oss:20b` digest below in Ollama. No runner downloads models. See [`experimental_configuration.json`](evaluation/results/experimental_configuration.json) for the recorded model, judge, seed, and retrieval settings.
- Obtain the 15 documents in [`public_corpus_manifest.json`](evaluation/benchmarks/public_corpus_manifest.json), with the listed filenames and exact SHA-256 hashes. Some sources require lawful publisher access. Raw PDFs are not distributed here. Use a separate public-only directory, never an email corpus or mixed index.
- Cache `NovaSearch/stella_en_1.5B_v5` revision `7817065102fd9e1b031fe874e910c01f40b2f001`, including its model code, and set `embedding_path` to that local snapshot directory, not the Hugging Face model name.
- Configure `CALC_API_KEY` and `GEOCODE_API_KEY` for calculator and named-place tools, and `OPENAI_API_KEY` for paid reference judging, through the environment or an untracked `.env`. These credentials are unnecessary for retrieval alone. Send only newly generated public-corpus answers to the external judge.

```bash
pip install -r agentic/requirements.txt -r evaluation/requirements.txt
run_dir=$(mktemp -d /tmp/chatgeomagi-eval.XXXXXX)
public_documents=/absolute/path/to/public_documents
embedding_path=/absolute/path/to/cached/stella/snapshot
model_digest=17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7
```

Build the index and score the 25-question retrieval benchmark (Hit@4, Recall@4, Precision@4 and annotated support coverage):

```bash
python evaluation/build_public_index.py \
  --manifest evaluation/benchmarks/public_corpus_manifest.json \
  --data-dir "$public_documents" --embedding-path "$embedding_path" \
  --index "$run_dir/index" --preset stella_s2p_cosine --k 4 \
  --benchmark evaluation/benchmarks/retrieval.json \
  --output "$run_dir/retrieval"
```

For the 14 held-out questions, reuse that index with `--skip-build`:

```bash
python evaluation/build_public_index.py \
  --manifest "$run_dir/index/indexed_manifest.json" \
  --data-dir "$public_documents" --embedding-path "$embedding_path" \
  --index "$run_dir/index" --preset stella_s2p_cosine --k 4 --skip-build \
  --benchmark evaluation/benchmarks/retrieval_heldout.json \
  --output "$run_dir/retrieval_heldout"
```

Run the paired 25-question RAG/no-RAG experiment with fresh sessions. Calculator and plotting tools remain available in both conditions:

```bash
python evaluation/run_revision_inference.py \
  --experiment ablation --execute-inference \
  --model gpt-oss:20b --model-digest "$model_digest" \
  --temperature 0 --seed 20261115 --num-ctx 8192 --num-predict 2048 \
  --manifest "$run_dir/index/indexed_manifest.json" \
  --data-dir "$public_documents" --embedding-path "$embedding_path" \
  --index "$run_dir/index" --preset stella_s2p_cosine --retrieval-k 4 \
  --output "$run_dir/ablation"
```

Judge the two conditions under distinct labels, explicitly supplying question, reference and answer. The judge runs calibration before scoring; each `--inputs` entry is `label=path:condition`:

```bash
python evaluation/openai_reference_judge.py \
  --model gpt-4o-mini --seed 20261115 \
  --inputs "no_rag=$run_dir/ablation/judge_inputs.json:no_rag" \
           "rag_15doc_corrected=$run_dir/ablation/judge_inputs.json:rag" \
  --output "$run_dir/judging"
```

Run the separate 20-prompt tool benchmark and score its saved traces against the checked-in independent NOAA references:

```bash
python evaluation/run_revision_inference.py \
  --experiment tools --execute-inference \
  --model gpt-oss:20b --model-digest "$model_digest" \
  --temperature 0 --seed 20261115 --num-ctx 8192 --num-predict 2048 \
  --output "$run_dir/tools"
python evaluation/score_live_tools.py \
  --traces "$run_dir/tools" --expected evaluation/benchmarks/tool_expected.json \
  --output "$run_dir/tool_scores"
```

`execution` checks successful tool calls and valid returned calculator data independently of strict arguments. `parameters_and_order` checks exact expected arguments and sequence, including January 1 endpoints. `tool_numerical_agreement` separately checks dates, values within tolerance, and returned model identity against NOAA references. The saved benchmark has **16/16 execution successes**, **12/16 strict parameter matches**, and **16/16 numerical matches**; T07–T09 and T15 used December 31 endpoints while returning the expected annual samples. The original invalid-request result remains **3/4**, including T19's coordinate-swap failure. The scorer leaves rejection decisions for separate trace review; the later coordinate-validation fix must not be counted as a historical success. These metrics do not establish end-to-end scientific correctness.

The commands reproduce the full-corpus workflow, not a guarantee of identical LLM outputs or present-day NOAA responses. Preserve the new traces, judge outputs and metadata separately. The published two-document sensitivity and older email experiment remain separate observations; they are not recreated by these commands.
