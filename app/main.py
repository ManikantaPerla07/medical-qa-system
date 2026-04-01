import json
import os

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# App configuration
app = FastAPI(
    title="Medical QA System",
    description="Domain-Specific Medical Question Answering using BioBERT",
    version="1.0.0",
)

MODEL_PATH = "./model/biobert_medical_qa"
id2label = {0: "no", 1: "yes", 2: "maybe"}

model = None
tokenizer = None


# Load model on startup
@app.on_event("startup")
async def load_model():
    """Load BioBERT tokenizer and model on application startup."""
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print("BioBERT model loaded successfully!")


# Request and Response schemas
class QARequest(BaseModel):
    """Schema for incoming QA requests."""

    question: str
    context: str


class QAResponse(BaseModel):
    """Schema for QA response with confidence scores."""

    question: str
    answer: str
    confidence: float
    context_used: str


class HealthResponse(BaseModel):
    """Schema for health check endpoint."""

    status: str
    model_loaded: bool


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check application health and model availability."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
    )


# Main QA endpoint
@app.post("/ask", response_model=QAResponse)
async def ask_question(request: QARequest):
    """Answer medical questions using BioBERT on provided context."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    # Tokenize question and context together
    inputs = tokenizer(
        request.question,
        request.context,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    # Run inference with torch.no_grad() for efficiency
    with torch.no_grad():
        logits = model(**inputs).logits

    # Compute probabilities and get predictions
    probabilities = torch.softmax(logits, dim=-1)
    pred_label_idx = torch.argmax(probabilities, dim=-1).item()
    confidence = float(probabilities[0, pred_label_idx].item())

    # Map to label string
    answer = id2label[pred_label_idx]

    return QAResponse(
        question=request.question,
        answer=answer,
        confidence=confidence,
        context_used=request.context[:200],
    )


# Run the app
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
