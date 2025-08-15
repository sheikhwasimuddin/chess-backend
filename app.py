# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd

# ------------------ artefacts ------------------
model = joblib.load("model/chess_model.pkl")
eco_enc = joblib.load("model/eco_encoder.pkl")
name_enc = joblib.load("model/name_encoder.pkl")
first_enc = joblib.load("model/first_move_encoder.pkl")

# ------------------ FastAPI --------------------
app = FastAPI(title="Chess ML Predictor")

# Allow Streamlit dev server (adjust if needed for Render)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for testing; tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ schemas --------------------
class PredictIn(BaseModel):
    white_rating: int
    black_rating: int
    moves_san: List[str]
    opening_eco: Optional[str] = None
    opening_name: Optional[str] = None

# ------------------ helpers --------------------
def safe_transform(encoder, value: Optional[str]) -> int:
    value = value or "unknown"
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return int(encoder.transform([encoder.classes_[0]])[0])

# ------------------ endpoint -------------------
@app.post("/predict")
def predict(body: PredictIn):
    feats = {
        "turns": len(body.moves_san),
        "white_rating": body.white_rating,
        "black_rating": body.black_rating,
        "rating_diff": body.white_rating - body.black_rating,
        "white_higher": int(body.white_rating > body.black_rating),
        "moves_count": len(body.moves_san),
        "opening_eco_enc": safe_transform(eco_enc, body.opening_eco),
        "opening_name_enc": safe_transform(name_enc, body.opening_name),
        "first_move_enc": safe_transform(
            first_enc, body.moves_san[0] if body.moves_san else None
        ),
        "opening_ply": min(len(body.moves_san), 8),
    }
    X = pd.DataFrame([feats])
    proba_white = float(model.predict_proba(X)[0, 1])
    return {"white_win_probability": proba_white}
