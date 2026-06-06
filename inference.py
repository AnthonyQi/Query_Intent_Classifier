from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="TheCupNoodle/query-intent-classifier",
    max_seq_length=2048,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

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
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()

if __name__ == "__main__":
    while True:
        q = input("Query: ")
        print(classify(q))
