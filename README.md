# ChatGeo-Magi

<a href="https://doi.org/10.5281/zenodo.21535623"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.21535623.svg" alt="DOI"></a>

This repository contains all the code of ChatGeo-Magi, including the agentic approach, two-LLM approach, evaluation, plotting, and the testing website.

ChatGeo-Magi is an AI agent for natural-language access to NOAA/CIRES geomagnetic data, models, and resources. It combines RAG with structured tool calling to answer questions, run API calculations, and generate visualizations.

## Running ChatGeo-Magi

Requires Python 3.11+, [Ollama](https://ollama.com) with the selected local model, and public geomagnetic documents in `agentic/data/`. Model and retrieval settings are in [`agentic/config.py`](agentic/config.py). Set `CALC_API_KEY` and `GEOCODE_API_KEY` in the environment or an untracked `.env` file for calculator and location queries.

```bash
pip install -r agentic/requirements.txt
cd agentic
python cgmterm.py
```

For the web interface, use `streamlit run cgmwebsite.py`.

## Evaluation Data

- [Results](evaluation/results/): retrieval, RAG/no-RAG, tool-calling, and London case-study results, including [experimental settings](evaluation/results/experimental_configuration.json).
- [Benchmarks](evaluation/benchmarks/) and [reference answers](evaluation/goldens.csv): questions, source annotations, and NOAA reference values.
- [Figures](evaluation/plot_outputs/): publication figures in PDF and PNG formats.
- [Historical email experiment](evaluation/legacy_email_experiment/): original email/no-email outputs. Publication permission for email-derived text remains subject to author/NOAA review.

## Reproducing Figures

If using Nix, you can simply use `nix develop` to enter a development environment with dependencies installed. Otherwise from the repository root:

```bash
pip install -r evaluation/requirements.txt
```

Then you can use `plots.py` to generate figures.

```bash
python evaluation/plots.py # Generates all of them
python evaluation/plots.py --figure rag # Specify a certain one
```

Other choices are `retrieval` and `questions`. Outputs are saved to `evaluation/plot_outputs/`.

## Rerunning Experiments

Install `agentic/requirements.txt` and `evaluation/requirements.txt`. Use the exact model and retrieval settings in the [experimental configuration](evaluation/results/experimental_configuration.json), a cached local Stella model, and the documents identified by the [public corpus manifest](evaluation/benchmarks/public_corpus_manifest.json). Some documents require publisher access. Model inference was done on a 40 GB A100 using Ollama, and reference judging additionally requires `OPENAI_API_KEY`.

The evaluation scripts provide their arguments through `--help`:
- [Retrieval evaluation](evaluation/build_public_index.py): build and evaluate the public-only index.
- [Model inference](evaluation/run_revision_inference.py): run RAG/no-RAG or tool-calling experiments.
- [Reference judging](evaluation/openai_reference_judge.py): score public-corpus answers against references.
- [Tool scoring](evaluation/score_live_tools.py): compare saved tool traces with independent NOAA references.
