# JHU 705.605 Introduction to Generative AI

Course materials and assignments for Johns Hopkins University's Introduction to Generative AI.

## Project structure

- **`modules/`** — Course modules (1–4) with assignments and Jupyter notebooks
- **`datasets/`** — Datasets used in assignments (Fashion MNIST, cats/dogs audio, time series, etc.)
- **`models/`** — Cached Hugging Face and other model artifacts
- **`conda-env.yml`** — Conda environment specification

## Conda environment

The project uses a Conda environment named `jhu-intro-to-genai` (Python 3.12, PyTorch, Transformers, Jupyter, etc.).

### Create the environment from the spec

```bash
conda env create -f conda-env.yml
```

### Activate the environment

```bash
conda activate jhu-intro-to-genai
```

### Update the environment after changes to `conda-env.yml`

```bash
conda env update -f conda-env.yml --prune
```

### Deactivate the environment

```bash
conda deactivate
```

### Export the current environment (e.g. after adding packages)

```bash
conda env export > conda-env.yml
```

For a more portable export (no build strings, only explicit specs):

```bash
conda env export --no-builds > conda-env.yml
```

### Remove the environment

```bash
conda env remove -n jhu-intro-to-genai
```

### List all conda environments

```bash
conda env list
```

## Running the notebooks

1. Activate the environment: `conda activate jhu-intro-to-genai`
2. Start Jupyter from the repo root: `jupyter notebook` or `jupyter lab`
3. Open the desired notebook under `modules/module N/`.

## Dependencies (from `conda-env.yml`)

- Python 3.12
- Jupyter, matplotlib, scikit-learn, nltk
- PyTorch, torchvision
- transformers, accelerate, safetensors, huggingface_hub
- tqdm, pmdarima
