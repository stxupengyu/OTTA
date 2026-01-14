# Scotta

Scotta is a research codebase for LLM-based multi-label classification with an optional RAG-style memory bank. It supports baseline and RAG modes and reports micro-F1, macro-F1, and example-F1.

## Requirements

- Python 3.10+
- Packages: `openai`, `numpy`, `sentence-transformers`

```bash
pip install openai numpy sentence-transformers
```

## Setup

Set API keys via environment variables (no keys are stored in the repo):

```bash
export OPENAI_API_KEY=YOUR_API_KEY
# Only if you use qwen2.5 via SiliconFlow:
export SILICONFLOW_API_KEY=YOUR_API_KEY
```

## Data Layout

Place datasets under `data/`:

```
data/
  MOVIE/
    test.txt
    tag_description.csv
  AAPD/
    test.txt
    tag_description.csv
  RCV/
    test.txt
    tag_description.csv
  StackExchange/
    test.txt
    tag_description.csv
```

`test.txt` format (blank line between samples):

```
Text: <text>
Labels: label1,label2
```

`tag_description.csv` format:

```
tags,outputs
label1,description for label1
```

## Run

Baseline mode:

```bash
python main.py --dataset movie --mode base --model-type gpt3.5
```

RAG mode:

```bash
python main.py --dataset movie --mode rag --model-type gpt3.5
```

Custom model name or API base URL:

```bash
python main.py --dataset movie --model-type qwen2.5 --base-url https://api.siliconflow.cn/v1
```

Limit samples for a quick run:

```bash
python main.py --dataset movie --max-samples 50
```

## Outputs

Results are written to `output/<DATASET>/`:

- `log.txt`: run logs
- `metrics.json`: final metrics
- `predictions.csv`: per-sample predictions

## Notes

- `--use-label-desc` adds label descriptions to the prompt.
- RAG mode performs extra API calls to estimate confidence for cache updates.
