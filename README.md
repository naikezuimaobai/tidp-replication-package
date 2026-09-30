# TIDP Experiment Reproduction Package

This package contains the experiment code, configuration templates, prompt templates, and reproduction instructions for the paper **Task-Intent-Driven Web Agent Task Decomposition and DOM Dynamic Pruning (TIDP)**.

## 1. Experimental environment

- Linux
- Python 3.11
- NVIDIA RTX A6000 40 GB, when GPU acceleration is used
- Sentence-BERT: `all-MiniLM-L6-v2`
- Cross-Encoder: `ms-marco-MiniLM-L-6-v2`
- Intent classifier: `Qwen2.5-3B`
- Action-decision models: GPT-4.1-mini or DeepSeek-v4-flash, according to the paper configuration
- Token-counting encoding: `cl100k_base`

## 2. Directory structure

```text
experiment_package/
├── linux/                         # Main experiment scripts and algorithm modules
├── single_step_intent_clustering/ # Intent clustering preprocessing and analysis
├── prompts/                       # Versioned prompt templates
├── configs/                       # Experiment configuration template
├── data/                          # Reserved for locally supplied datasets
├── results/                       # Generated CSV/JSON results
├── figures/                       # Generated plots
├── tests/                         # Static and smoke tests
├── requirements.txt               # Unpinned development dependencies
└── requirements-lock.txt          # Pinned application-level dependencies
```

## 3. Data and model preparation

The package does not redistribute the original Mind2Web data, model weights, or API keys. The required resources can be obtained from the following sources:

- Mind2Web dataset: [official project page](https://osu-nlp-group.github.io/Mind2Web/)
- Sentence-BERT model (`all-MiniLM-L6-v2`): [Hugging Face mirror](https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/tree/main)

Prepare the following resources locally:

1. Mind2Web, with its root path set in `configs/experiment.yaml` or supplied through the command line;
2. `all-MiniLM-L6-v2` and `ms-marco-MiniLM-L-6-v2`;
3. `qwen2.5:3b` in a local Ollama service, or the remote model endpoint used in the paper;
4. API credentials for remote action-decision models, supplied through environment variables rather than source files.

## 4. Installation

Use a Python 3.11 virtual environment:

```bash
python -m pip install -r requirements-lock.txt
```

The PyTorch wheel must match the CUDA driver on the target machine. For strict reproduction, record the CUDA, PyTorch, model-service, and dataset versions together with the experiment results.

## 5. Running the experiments

Run commands from the package root:

```bash
python linux/experiment1.py --split test_cross_task
python linux/experiment1.py --split test_cross_website
python linux/experiment1.py --split test_cross_domain
```

Run the ablation experiment with:

```bash
python linux/experiment1_ablation.py --split test_cross_task
```

The intent-clustering scripts are located in `single_step_intent_clustering/`. Set `TIDP_CLUSTER_BASE_PATH` to the directory containing the clustering inputs and generated intermediate files before running them.

## 6. Method naming and legacy result keys

All user-facing documentation and output labels use **TIDP**. Some CSV/JSON fields and filenames retain the historical `icddp` key for compatibility with existing result-processing scripts; this key does not denote a different method.

## 7. Reproduction notes

- Check the dataset split, random seed, context limit, and token encoding before each run.
- Generated results and figures are written to `results/` and `figures/` when the corresponding scripts are run from the package root.
- Remote model responses may vary with service and model versions. Save the effective configuration and raw outputs used for reported results.
- The threshold policy and hierarchical state-handling logic from the current paper implementation are preserved.

## 8. License

The code and documentation are distributed under the license in `LICENSE`. The dataset and third-party models remain subject to their own licenses.
