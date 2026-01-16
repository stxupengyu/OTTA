# SCOTTA

**SCOTTA** is an implementation of a training-free online test-time adaptation framework for
LLM-based multi-label text classification.

## Overview

SCOTTA enables large language models (LLMs) to adapt to streaming test data without updating
model parameters. The framework includes two core components:

1. **Label-set Local Likelihood Ratio (L3R)**: a decoding-consistent confidence metric that
   models label competition by focusing on critical branching points.
2. **Submodular Memory Bank (SMB)**: a cache maintenance strategy that balances coverage,
   semantic diversity, and sample quality under a fixed context budget.

## Installation

### Requirements

- Python 3.10+
- Dependencies: `openai`, `numpy`, `sentence-transformers`

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY=your_openai_api_key
export SILICONFLOW_API_KEY=your_siliconflow_api_key
```

## Data Format

Dataset download link: <https://www.dropbox.com/scl/fo/4lgjy1an6jmi97k0vb2ec/ALvBHS-RwUaK-QEsVltiQ5E?rlkey=kxbl3eflgyg2m4exjofzm5qmp&e=1&st=dj9ufgtp&dl=0>

Place datasets under `data/` with the following structure:

```
data/
  MOVIE/
  AAPD/
  RCV/
  StackExchange/
```

## Quick Start

### Basic usage

**Static LLM inference (baseline):**

```bash
python main.py \
    --dataset movie \
    --mode base \
    --model-type gpt3.5
```

**SCOTTA (full framework):**

```bash
python main.py \
    --dataset movie \
    --mode rag \
    --model-type gpt3.5 \
    --bank-type smb \
    --use-label-desc \
    --conf-type l3r
```

**Run all datasets (paper hyperparameters):**

```bash
bash scripts/run.sh [model_type]  # default: gpt3.5
```

## Command-line Arguments

### Dataset

- `--dataset`: dataset name
  - values: `movie`, `aapd`, `rcv`, `se` (StackExchange)
  - default: `movie`

### Mode

- `--mode`: experiment mode
  - `base`: static LLM inference (no adaptation)
  - `rag`: SCOTTA online test-time adaptation
  - default: `base`

### Model

- `--model-type`: LLM backbone
  - values: `gpt3.5`, `gpt4o`, `qwen2.5`
  - default: `gpt3.5`
- `--model-name`: custom model name (overrides `--model-type`)
- `--api-key`: API key (or use env vars)
- `--base-url`: API base URL (auto-selected by model type)
- `--system-prompt`: custom system prompt (default enforces JSON output)
- `--use-label-desc`: include label descriptions in the prompt
- `--use-dataset-desc`: include dataset description in the prompt

### Memory bank

- `--bank-type`: memory bank type
  - `naive`: FIFO replacement by confidence
  - `smb`: submodular memory bank (recommended)
  - default: `naive`
- `--cache-size`: memory capacity (naive and smb, default: `100`)
- `--rag-k`: RAG retrieval top-k (default: `10`)
- `--smb-lambda1`: SMB coverage weight (default: `1.0`)
- `--smb-lambda2`: SMB diversity weight (default: `1.0`)
- `--smb-lambda3`: SMB quality weight (default: `1.0`)
- `--smb-epsilon`: SMB coverage epsilon (default: `1e-12`)
- `--smb-W`: SMB candidate_pool window size (default: `200`)

### L3R confidence

- `--conf-type`: confidence type
  - `naive`: token-probability confidence
  - `l3r`: L3R confidence (recommended)
- `--tau`: confidence threshold for cache updates
- `--l3r-eps`: L3R smoothing constant (default: `1e-12`)
- `--l3r-alpha`: L3R softmax sharpness (default: `1.0`)
- `--l3r-agg`: instance confidence aggregation
  - `mean`: average label confidence (recommended)
  - `max`: max label confidence
  - `top-m`: mean of top-m label confidences
- `--l3r-top-m`: m for top-m aggregation
- `--l3r-validate`: print one-time L3R validation info

### RAG mode

- `--rag-warmup`: warmup samples before RAG retrieval (default: `200`)
- `--encoder-model`: encoder model for text embeddings (default: `sentence-transformers/all-mpnet-base-v2`)
- `--se-candidate-topk`: SE stage-1 candidate labels (default: `100`)
- `--se-label-embed-batch-size`: SE label embedding batch size (default: `256`)

### Evaluation

- `--max-samples`: limit test samples (for quick runs)
- `--request-interval`: sleep between API calls (seconds)

### Output

- `--output-dir`: output directory (default: `output/{dataset}/`)

## Hyperparameters

Recommended defaults from the paper:

| Dataset        | B (capacity) | k (retrieval) | tau (threshold) | (lambda1, lambda2, lambda3) |
| -------------- | ------------ | ------------- | --------------- | --------------------------- |
| MOVIE          | 64           | 4             | 0.80            | (0.35, 0.35, 0.30)          |
| AAPD           | 128          | 6             | 0.72            | (0.40, 0.30, 0.30)          |
| RCV1           | 256          | 6             | 0.70            | (0.40, 0.30, 0.30)          |
| StackExchange  | 512          | 8             | 0.62            | (0.50, 0.25, 0.25)          |

**L3R parameters** (shared across datasets):

- `--l3r-eps`: `1e-8`
- `--l3r-alpha`: `5.0`

When using `--bank-type smb` without custom overrides, these defaults are applied.

**Batch script**: `scripts/run.sh` applies the paper hyperparameters to all datasets.

## Outputs

Results are saved under `output/{dataset}/`:

- `log.txt`: detailed runtime logs
- `metrics.json`: final metrics

  ```json
  {
    "micro-f1": 71.86,
    "macro-f1": 59.15,
    "example-f1": 60.23
  }
  ```

- `predictions.csv`: per-sample predictions

  - `id`: sample index
  - `text`: input text
  - `gold_labels`: gold labels (semicolon-separated)
  - `predicted_labels`: predicted labels (semicolon-separated)
