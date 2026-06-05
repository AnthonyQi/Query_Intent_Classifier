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
```

```text
compare UCLA and UC Berkeley engineering
```

```text
best colleges for computer science in California
```

```text
how much is tuition at Purdue
```

The objective is to classify each query into structured labels that can later be used for routing and decision-making.

Outputs include:

* Persona
* Intent
* Intent Details

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

## Dataset Creation

### Source Data

The dataset originated from a labeled CSV file:

```text
data/datasheet.csv
```

Each row contains:

* Query text
* Persona label
* Intent label
* Additional metadata

### Training Format

The CSV data was converted into instruction-style examples for supervised fine-tuning.

Example:

```json
{
  "instruction": "Classify this college-related query.",
  "input": "schools like MIT but cheaper",
  "output": "persona: high_school_student\nintent: recommendation_filtering\nintent_details: comparison_search"
}
```

The resulting dataset is stored in:

```text
data/train_data.json
```

---

## Model Training

Training was performed using:

* Unsloth
* LoRA fine-tuning
* Qwen2.5-3B-Instruct-bnb-4bit

The model was trained to map college-related queries to the appropriate persona and intent labels.

---

## Repository Structure

```text
.
├── data/
│   ├── datasheet.csv
│   └── train_data.json
│
├── final_model/
│   ├── adapter_config.json
│   ├── chat_template.jinja
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── inference.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Model Files

The repository includes:

* Training dataset
* LoRA configuration
* Tokenizer files
* Inference code

The trained adapter weights file:

```text
adapter_model.safetensors
```

is not included in this repository because it exceeds GitHub's 100 MB file size limit.

---

## Running Inference

Install dependencies:

```bash
pip install -r requirements.txt
```

Run inference:

```bash
python inference.py
```

Example query:

```text
schools like MIT but cheaper
```

Example output:

```text
persona: high_school_student
intent: recommendation_filtering
intent_details: comparison_search
```

---

## Current Status

Completed:

* Dataset generation and labeling
* CSV-to-training-format conversion
* LoRA fine-tuning with Unsloth
* Model export
* Inference pipeline
* GitHub repository setup

Next Steps:

* Validation on unseen queries
* Accuracy evaluation
* Intent confusion analysis
* Taxonomy refinement
* Routing strategy development

---

## Project Context

This project is part of a larger query-routing system designed to:

1. Understand user intent.
2. Identify user persona.
3. Route simple queries directly to search.
4. Route complex queries to more advanced workflows.

The classifier serves as the first stage of that routing pipeline.
