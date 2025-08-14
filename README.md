# Autoregressive Text Generation with GPT-2

This project demonstrates **autoregressive text generation** using OpenAI's GPT-2 model.  
It takes an initial prompt (seed text) and generates coherent continuations using the model’s learned language patterns.

---

## Features
- Uses **Hugging Face Transformers** for easy GPT-2 integration.
- Generates text based on **autoregressive language modeling**.
- Supports **custom prompt input** from the user.
- Adjustable **generation length, temperature, and top-k/top-p sampling** for creative control.
- Minimal and easy-to-understand code structure.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Shreyash-Dayma/autoregressive-text-gen-gpt2.git
   cd autoregressive-text-gen-gpt2

2.```bash

    python -m venv venv
    source venv/bin/activate   # On Windows use: venv\Scripts\activate
    
3.  Run the script with a custom prompt:
```bash

  python generate.py --prompt "Once upon a time"

## What's Happening Here

> **Prompt:** “Once upon a time…”
>
> **Tokenizer:** Converts words to numbers.  
> **GPT-2:** Predicts next word one at a time, adds it to the sentence, then predicts again — autoregressive generation.  
> **Stops:** When `max_length` is reached.  
> **Final Step:** Converts numbers back to text.
