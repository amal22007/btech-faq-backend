from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Initialize FastAPI app
app = FastAPI()

# ✅ Allow CORS for all origins (you can restrict this later for security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model and tokenizer
model_path = "btech-faq-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# ✅ Labels and corresponding answers (customize as needed)
labels = ['admission_process', 'cutoff', 'fees', 'placements', 'facilities', 'ranking', 'courses', 'management_quota', 'top_colleges', 'hostel']
label_to_answer = {
    'admission_process': "You can apply through the university portal.",
    'cutoff': "The cutoff varies based on category and college.",
    'fees': "The average BTech fee ranges from ₹50,000 to ₹1,50,000 per year.",
    'placements': "Most top colleges offer good placement opportunities.",
    'facilities': "Facilities include labs, libraries, Wi-Fi, and sports.",
    'ranking': "Refer to NIRF rankings or college-specific accreditations.",
    'courses': "Popular courses include CSE, ECE, ME, CE, IT, and AI/DS.",
    'management_quota': "Yes, management quota is available in many colleges.",
    'top_colleges': "Top colleges include CET Trivandrum, NIT Calicut, TKM, etc.",
    'hostel': "Most colleges offer hostel facilities for both boys and girls.",
}

# ✅ Pydantic schema
class Query(BaseModel):
    question: str

# ✅ Prediction endpoint
@app.post("/predict")
def predict(data: Query):
    inputs = tokenizer(data.question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_index = torch.argmax(probs).item()
        pred_label = labels[pred_index]
        answer = label_to_answer.get(pred_label, "Sorry, I don't have an answer for that.")
    return {"answer": answer}
