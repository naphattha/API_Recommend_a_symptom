# api.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from symptom_recommendation import predict_from_row  # your function

app = FastAPI()

# === Define expected input format ===
class PatientInput(BaseModel):
    gender: str
    age: int
    summary: str
    search_term: Optional[str] = ""

@app.post("/recommend")
def recommend_symptoms_api(patient: PatientInput):
    row_dict = patient.dict()
    recommendations = predict_from_row(row_dict)
    return {"recommended_symptoms": recommendations}

# === Optional: run using python api.py ===
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
