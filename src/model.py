"""
Prediction wrapper for the trained XGBoost emotion classifier.

Loads the DistilBERT model once and reuses it across predictions, rather than
reloading it per call. 

Usage:
    from model import EmotionClassifier
    clf = EmotionClassifier()
    clf.predict("stocks plunge after interest rate hike")
"""

import joblib
import numpy as np
import pandas as pd

from preprocess import (
    NRC_EMOTIONS,
    load_distilbert_model,
    get_distilbert_embedding,
    extract_affect_frequencies,
    ensure_nltk_resources,
)

MODEL_PATH = "models/xgb_distilbert_nrc.joblib"


class EmotionClassifier:
    def __init__(self, model_path: str = MODEL_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.label_encoder = bundle["label_encoder"]

        ensure_nltk_resources()
        self.tokenizer, self.distilbert_model, self.device = load_distilbert_model()

        #Column order must match training: NRC emotions, then DistilBERT dimensions (see preprocess.combine_features)
        self.feature_columns = NRC_EMOTIONS + [
            f"distilbert_dim_{i+1}" for i in range(768)
        ]

    def _extract_features(self, text: str) -> pd.DataFrame:
        """Build a single-row feature vector for one headline, in training column order."""
        nrc_scores = extract_affect_frequencies(text)
        distilbert_vector = get_distilbert_embedding(
            text, self.tokenizer, self.distilbert_model, self.device
        )

        row = {emotion: nrc_scores[emotion] for emotion in NRC_EMOTIONS}
        row.update({
            f"distilbert_dim_{i+1}": value
            for i, value in enumerate(distilbert_vector)
        })

        return pd.DataFrame([row], columns=self.feature_columns)

    def predict(self, text: str) -> dict:
        """Predict the emotion label and per-class probabilities for one headline."""
        features = self._extract_features(text)

        pred_idx = self.model.predict(features)[0]
        probs = self.model.predict_proba(features)[0]

        label = self.label_encoder.inverse_transform([pred_idx])[0]
        class_probs = {
            cls: float(prob)
            for cls, prob in zip(self.label_encoder.classes_, probs)
        }

        return {"label": label, "probabilities": class_probs}


if __name__ == "__main__":
    clf = EmotionClassifier()
    example = "stocks plunge after interest rate hike"
    result = clf.predict(example)
    print(f"Text: {example}")
    print(f"Predicted emotion: {result['label']}")
    print(f"Probabilities: {result['probabilities']}")
