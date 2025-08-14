from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. Load the pretrained model and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 2. Example sentence in Spanish / Italian (change for your test)
sentence = "Questo ristorante è fantastico, il cibo e il servizio sono eccellenti."  # Spanish: "I love this product, it's fantastic"

# 3. Tokenize and prepare the input
inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)

# 4. Run the model
with torch.no_grad():
    outputs = model(**inputs)

# 5. Get predicted class (model outputs logits for 1–5 stars)
predicted_star = torch.argmax(outputs.logits, dim=1).item() + 1

print(f"Sentence: {sentence}")
print(f"Predicted Sentiment: {predicted_star} star(s)")
