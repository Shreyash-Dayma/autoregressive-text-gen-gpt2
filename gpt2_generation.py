from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# 1. Load model & tokenizer
model_name = "gpt2"  # you can try "gpt2-medium" or "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 2. Prompt for generation
prompt = "Once upon a time in a small village"

# 3. Tokenize input
inputs = tokenizer.encode(prompt, return_tensors="pt")

# 4. Generate text
outputs = model.generate(
    inputs,
    max_length=50,        # total length (prompt + generated)
    num_return_sequences=1, # number of outputs
    temperature=0.7,      # creativity level (lower = safer, higher = more random)
    top_p=0.9,            # nucleus sampling
    do_sample=True        # random sampling instead of greedy
)

# 5. Decode & print
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:\n", generated_text)
