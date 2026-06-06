# College Query Classifier

## Overview

This project fine-tunes a language model to classify college-related queries into structured intent and persona categories.

The goal is to support a larger query routing system that can determine what a user is asking and route them to the appropriate search, recommendation, or advisory workflow.

Base Model:
```text
unsloth/Qwen2.5-3B-Instruct-bnb-4bit
```

Fine-Tuning Method:
```text
LoRA + Unsloth
```

---

## Problem Definition

College-related queries vary significantly in complexity and intent.

Examples:
```text
schools like MIT but cheaper
compare UCLA and UC Berkeley engineering
best colleges for computer science in California
how much is tuition at Purdue
```

The objective is to classify each query into structured labels that can later be used for routing and decision-making.

Outputs include:
- Persona
- Intent
- Intent Details

Example:
```text
Input:
schools like MIT but cheaper

Output:
persona: high_school_student
intent: recommendation_filtering
intent_details: comparison_search
```

---

## Requirements

- Python 3.10+
- A GPU (CUDA-required). If you don't have one locally, use [Google Colab](https://colab.research.google.com/) with a T4 GPU runtime.
- A [Hugging Face](https://huggingface.co) account (required to download the private model weights)

---

## Setup & Running Inference

### 1. Clone the repo
```bash
git clone https://github.com/AnthonyQi/Query_Intent_Classifier.git
cd Query_Intent_Classifier
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Log into Hugging Face
The model weights are hosted privately on Hugging Face. You need an account and access token to download them.

- Sign up at [huggingface.co](https://huggingface.co)
- Go to **Settings → Access Tokens → New Token**
- Set role to **Read**, copy the `hf_...` token
- Run:
```bash
pip install huggingface_hub
hf auth login
# paste your token when prompted
```

> Ask to add your HF username to the private repo if you get an access error.

### 5. Run inference
```bash
python inference.py
```

You'll get an interactive prompt:
```text
Query: schools like MIT but cheaper
persona: high_school_student
intent: recommendation_filtering
intent_details: comparison_search
```

---

## Running on Google Colab (no local GPU)

1. Open a new Colab notebook and set runtime to **T4 GPU** (Runtime → Change runtime type)
2. Clone the repo and install deps:
```python
!git clone https://github.com/AnthonyQi/Query_Intent_Classifier.git
%cd Query_Intent_Classifier
!pip install -r requirements.txt
!pip install huggingface_hub
```
3. Log into Hugging Face:
```python
from huggingface_hub import login
login()  # paste your hf_... token
```
4. Run inference:
```python
exec(open("inference.py").read())
```

---

## Repository Structure

```text
.
├── data/
│   ├── datasheet.csv          # labeled query dataset
│   └── train_data.json        # formatted training data
│
├── final_model/
│   ├── adapter_config.json    # LoRA config
│   ├── chat_template.jinja    # chat template
│   ├── tokenizer.json         # tokenizer
│   └── tokenizer_config.json  # tokenizer config
│
├── inference.py               # run this to classify queries
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Model Files

Adapter weights (`adapter_model.safetensors`) are hosted on Hugging Face at:
```text
TheCupNoodle/query-intent-classifier
```

They are downloaded automatically when you run `inference.py` after logging in with `hf auth login`.

---

## Dataset Creation

### Source Data

The dataset originated from a labeled CSV file (`data/datasheet.csv`). Each row contains:
- Query text
- Persona label
- Intent label
- Additional metadata

### Training Format

The CSV data was converted into instruction-style examples for supervised fine-tuning:

```json
{
  "instruction": "Classify this college-related query.",
  "input": "schools like MIT but cheaper",
  "output": "persona: high_school_student\nintent: recommendation_filtering\nintent_details: comparison_search"
}
```

The resulting dataset is stored in `data/train_data.json`.

---

## Model Training

Training was performed using:
- Unsloth
- LoRA fine-tuning
- Qwen2.5-3B-Instruct-bnb-4bit

---

## Current Status

Completed:
- Dataset generation and labeling
- CSV-to-training-format conversion
- LoRA fine-tuning with Unsloth
- Model export and Hugging Face upload
- Inference pipeline
- GitHub repository setup

Next Steps:
- Validation on unseen queries
- Accuracy evaluation
- Intent confusion analysis
- Taxonomy refinement
- Routing strategy development

---

## Project Context

This project is part of a larger query-routing system designed to:
1. Understand user intent
2. Identify user persona
3. Route simple queries directly to search
4. Route complex queries to more advanced workflows

The classifier serves as the first stage of that routing pipeline.
