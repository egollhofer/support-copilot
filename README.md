# Support Copilot

This repository contains a small Python script that demonstrates a hybrid conversational support workflow using the Concentrate API.

The script takes a customer email and internal help documentation as input, then generates a customer-ready email response using multiple LLM providers.

---

## Requirements

* Python 3.9+
* A Concentrate API key

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

```bash
export CONCENTRATE_API_KEY=your_api_key_here
```

---

## Running the Script

```bash
python support_copilot.py \
  --email examples/refund_request.txt \
  --kb kb/
```

---

## Inputs

* `examples/`
  Contains sample customer emails.

* `kb/`
  Contains markdown files representing internal help documentation.

---

## Outputs

When the script runs, it creates an `outputs/` directory containing:

* `answer_draft.json` — initial grounded support answer
* `review.json` — model-generated review and feedback
* `email_final.txt` — final customer-ready email
* `run_log.jsonl` — raw API responses and metadata

---

## Notes

* The script uses multiple LLM providers through the Concentrate API.
* Model names can be changed via command-line arguments.
* Outputs are regenerated on each run.

