# College Query Classifier

## Qwen2.5 Intent Classifier (LoRA + Unsloth)

This project fine-tunes a lightweight language model to classify college-related queries into structured intent labels using LoRA and Unsloth.

Base model: `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`

---

## 1. Problem Definition

The goal is to classify college-related user queries into structured intent categories such as:
- recommendation filtering
- comparison requests
- tuition or attribute lookup
- campus life questions

---

## 2. Dataset Creation (`train_data.json`)

### Source

The dataset was originally derived from a structured CSV file (`datasheet.csv`) containing:
- user query text
- corresponding intent labels and metadata

### Conversion

Each row was converted into instruction-style format for supervised fine-tuning:

```json
{
  "instruction": "Classify this college-related query.",
  "input": "schools like MIT but cheaper",
  "output": "persona: counselor_teacher\nintent: recommendation_filtering\nintent_details: comparison_search"
}
