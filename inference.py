from unsloth import FastLanguageModel
import torch

# Load base model (same one you trained on)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Load LoRA adapter
model = FastLanguageModel.get_peft_model(model)

# Example inference function
def classify(query):
    prompt = f"""### Instruction:
Classify this college-related query.

### Input:
{query}

### Response:"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=50
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    while True:
        q = input("Query: ")
        print(classify(q))
      
