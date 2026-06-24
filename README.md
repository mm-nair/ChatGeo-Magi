# ChatGeo-Magi

[![DOI](https://zenodo.org/badge/1268798117.svg)](https://doi.org/10.5281/zenodo.20821922)

This repository contains all the code of ChatGeo-Magi, including the agentic approach, two-LLM approach, evaluation, plotting, and the testing website.

ChatGeo-Magi is an AI agent for natural-language access to NOAA/CIRES geomagnetic data, models, and resources. It combines RAG with structured tool calling to answer questions, run API calculations, and generate visualizations.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) for the local models (`llama3.3:70b`, `mistral-nemo:12b`, `gpt-oss:20b`)
- An OpenAI API key for the Opik judge (only needed for evaluation)
- Data (such as PDFs or text files) added to agentic/data/ to form the knowledge base

## Inferencing

To get the system running, simply install the dependencies and start it up:
```
cd agentic

# add API keys in .env AND within agentic/apis/noaa_apis.py
# choose model settings in config.py
# populate data/

# it is highly recommended to create a virtual environment to avoid package conflicts
# python -m venv venv
# source venv/bin/activate

pip install -r requirements.txt
python cgmterm.py # for a simple terminal interface
streamlit run cgmwebsite.py # for a website interface
```

## Reproducing Evaluation

Evaluation is done in 4 steps.
1. Generate goldens
2. Run inference in all conditions needed
3. Run the Opik evaluation on the inference results, and the "golden" answers
4. Download from Opik (or if done locally, move files) and run plotting scripts
```
mv evaluation/goldens.py agentic/goldens.py
cd agentic
python goldens.py

cd ..

mv evaulation/eval_inference.py agentic/eval_inference.py
cd agentic
python eval_inference.py

# move the output CSVs to evaluation/inference_results/

cd ..
cd evaluation
python eval_opik.py

# download or move results to evaluation/opik_evaluation_results/

python plots_1.py
python plots_2.py

# resulting plots are now in evaluation/plot_outputs/
```

## Notes
- The hallucination metric is included in the plotting and evaluation, but insignificant and thus omitted from the paper.