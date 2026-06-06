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

Example:
```text
Input:
schools like MIT but cheaper

Output:
persona: high_school_student
intent: recommendation
```

---

## Valid Labels

### Personas
| Label | Description |
|---|---|
| `high_school_student` | High school student researching colleges |
| `college_student` | Current college student |
| `parent` | Parent of a prospective student |
| `advisor` | Academic or independent advisor |
| `counselor_teacher` | School counselor or teacher |
| `college_b2b` | College or university (B2B) |
| `transfer_student` | Transfer student |
| `community_college_student` | Community college student |
| `graduate_applicant` | Graduate school applicant |
| `international_student` | International student |
| `career_changer` | Career changer returning to education |
| `edtech_founder` | EdTech founder or operator |
| `industry_hiring_manager` | Industry hiring manager |
| `scholarship_officer` | Scholarship foundation officer |
| `employer_recruiter` | Employer or recruiter |
| `government_analyst` | Government workforce analyst |

### Intents
| Label | Description |
|---|---|
| `exact_lookup` | Looking up a specific school by name |
| `attribute_lookup` | Looking up a specific attribute (e.g. tuition, ranking) |
| `filtered_search` | Searching with 1-2 filters |
| `multi_constraint` | Searching with multiple constraints |
| `comparison` | Comparing two or more schools |
| `recommendation` | Asking for school recommendations |
| `advisory` | Seeking general advice or guidance |
| `emotional_advisory` | Seeking emotional support or reassurance |
| `admissions_process` | Questions about the admissions process |
| `career_outcomes` | Questions about career outcomes or job placement |
| `cost_financial_aid` | Questions about cost or financial aid |
| `campus_life_fit` | Questions about campus life or culture fit |
| `b2b_partnership` | B2B or institutional partnership queries |
| `strategy` | Strategic planning queries |
| `reflective_advisory` | Reflective or identity-based advisory |
| `analytics_request` | Requests for data or analytics |
| `profile_management` | Profile or account management |

---

## Requirements

- Python 3.10+
- A GPU (CUDA required). If you don't have one locally, use [Google Colab](https://colab.research.google.com/) with a T4 GPU runtime.
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
- Go to **Settings -> Access Tokens -> New Token**
- Set role to **Read**, copy the `hf_...` token
- Run:
```bash
pip install -r requirements.txt
python inference.py
# paste your token when prompted
```

> Ask Anthony to add your HF username to the private repo if you get an access error.

### 5. Run inference
```bash
python inference.py
```

You'll get an interactive prompt:
```text
Query: schools like MIT but cheaper
persona: high_school_student
intent: recommendation

Query: UCLA tuition
persona: parent
intent: attribute_lookup
```

---

## Running on Google Colab (no local GPU)

1. Open a new Colab notebook and set runtime to **T4 GPU** (Runtime -> Change runtime type)
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
│   ├── datasheet.csv              # original labeled query dataset
│   ├── train_data.json            # initial training data
│   └── train_data_clean_v2.json   # cleaned dataset used for retraining
│
├── final_model/
│   ├── adapter_config.json        # LoRA adapter config
│   ├── adapter_model.safetensors  # trained weights (hosted on HF, not in repo)
│   ├── chat_template.jinja        # chat template
│   ├── tokenizer.json             # tokenizer
│   └── tokenizer_config.json      # tokenizer config
│
├── inference.py                   # run this to classify queries
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

They are downloaded automatically when you run `inference.py` after logging in with `hf auth login`. The file exceeds GitHub's 100MB limit and is excluded from this repo via `.gitignore`.

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
  "output": "persona: high_school_student\nintent: recommendation"
}
```

The cleaned dataset used for retraining is stored in `data/train_data_clean_v2.json` (9,252 examples with normalized labels).

---

## Model Training

Training was performed using:
- Unsloth
- LoRA fine-tuning
- Qwen2.5-3B-Instruct-bnb-4bit
- Two training runs: initial fine-tune + retrain on cleaned labels

---

## Current Status

Completed:
- Dataset generation and labeling
- CSV-to-training-format conversion
- Initial LoRA fine-tuning with Unsloth
- Label cleaning and normalization (v2)
- Retrain on cleaned dataset
- Model export and Hugging Face upload
- Inference pipeline with label validation
- GitHub repository setup

Next Steps:
- Validation on 100-200 unseen queries
- Accuracy measurement per intent and persona
- Routing tier mapping
- Taxonomy refinement based on validation results

---

## Project Context

This project is part of a larger query-routing system designed to:
1. Understand user intent
2. Identify user persona
3. Route simple queries directly to search
4. Route complex queries to more advanced workflows

The classifier serves as the first stage of that routing pipeline.
