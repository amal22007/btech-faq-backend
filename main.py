from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

app = FastAPI()

# Load model and tokenizer
model_path = "btech-faq-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Replace with your actual labels
labels = ['admission_process', 'cutoff', 'fees', 'placements', 'facilities', 'ranking', 'courses', 'management_quota', 'top_colleges', 'hostel']

class Query(BaseModel):
    question: str

@app.post("/predict")
def predict(data: Query):
    inputs = tokenizer(data.question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_label = labels[torch.argmax(probs).item()]
    return {"answer": pred_label}
