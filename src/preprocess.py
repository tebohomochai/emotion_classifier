"""
Feature extraction for the emotion classifier.

Builds a combined NRC + DistilBERT feature matrix from the raw headline
dataset. Each expensive step is cached to disk so re-running this script
does not recompute embeddings/features that already exist.

Usage:
    python src/preprocess.py
"""

import os

import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
from nrclex import NRCLex
import nltk

#Config
RAW_DATA_PATH = "data/final_augmented_dataset.csv"
DISTILBERT_FEATURES_PATH = "data/distilbert_features.csv"
NRC_FEATURES_PATH = "data/nrc_features.csv"
COMBINED_FEATURES_PATH = "data/nrc_distilbert_features.csv"

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64

NRC_EMOTIONS = [
    "fear", "anger", "anticipation", "trust",
    "surprise", "positive", "negative", "sadness",
    "disgust", "joy",
]


#Load data 
def load_dataset(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the augmented headlines dataset (semicolon-delimited)."""
    df = pd.read_csv(path, sep=";")
    print(f"Dataset shape: {df.shape}")
    return df


#Generate DistilBERT features
def load_distilbert_model(model_name: str = MODEL_NAME):
    """Load tokenizer + model, moved to GPU if available."""
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Using device: {device}")
    return tokenizer, model, device


def get_distilbert_embedding(text, tokenizer, model, device, max_length: int = MAX_LENGTH) -> np.ndarray:
    """Tokenise one headline and return its [CLS] embedding from the last hidden state."""
    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()


def extract_distilbert_features(df: pd.DataFrame, cache_path: str = DISTILBERT_FEATURES_PATH) -> pd.DataFrame:
    """Extract DistilBERT CLS embeddings for every headline, or load from cache."""
    if os.path.exists(cache_path):
        print(f"Found cached DistilBERT features at {cache_path}, skipping extraction.")
        return pd.read_csv(cache_path)

    tokenizer, model, device = load_distilbert_model()

    features = np.array([
        get_distilbert_embedding(text, tokenizer, model, device)
        for text in tqdm(df["text"], desc="Extracting DistilBERT embeddings")
    ])
    print(f"DistilBERT feature matrix shape: {features.shape}")

    distilbert_df = pd.DataFrame(
        features,
        columns=[f"distilbert_dim_{i+1}" for i in range(features.shape[1])],
    )
    distilbert_df["label"] = df["label"].values
    distilbert_df.to_csv(cache_path, index=False)
    print(f"Saved DistilBERT features to {cache_path}")
    return distilbert_df


#Generate NRC features
def ensure_nltk_resources():
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)


def extract_affect_frequencies(text) -> dict:
    """Return the NRC affect-frequency score for each emotion, defaulting to 0.0."""
    emotion = NRCLex(str(text))
    emotion.load_raw_text(str(text))
    scores = emotion.affect_frequencies
    return {e: scores.get(e, 0.0) for e in NRC_EMOTIONS}


def extract_nrc_features(df: pd.DataFrame, cache_path: str = NRC_FEATURES_PATH) -> pd.DataFrame:
    """Extract NRC proportion-based affect features for every headline, or load from cache."""
    if os.path.exists(cache_path):
        print(f"Found cached NRC features at {cache_path}, skipping extraction.")
        return pd.read_csv(cache_path)

    ensure_nltk_resources()

    nrc_df = df["text"].apply(extract_affect_frequencies).apply(pd.Series)
    nrc_df["label"] = df["label"].values

    all_zero = (nrc_df[NRC_EMOTIONS] == 0).all(axis=1).sum()
    print(f"NRC feature shape: {nrc_df.shape} | all-zero rows: {all_zero}")

    nrc_df.to_csv(cache_path, index=False)
    print(f"Saved NRC features to {cache_path}")
    return nrc_df


#Concatenate the features
def combine_features(
    nrc_df: pd.DataFrame,
    distilbert_df: pd.DataFrame,
    output_path: str = COMBINED_FEATURES_PATH,
) -> pd.DataFrame:
    """Concatenate NRC affect features with DistilBERT embeddings into one matrix."""
    nrc_only = nrc_df[NRC_EMOTIONS]
    combined = pd.concat(
        [nrc_only.reset_index(drop=True), distilbert_df.drop(columns=["label"]).reset_index(drop=True)],
        axis=1,
    )
    combined["label"] = nrc_df["label"].values
    combined.to_csv(output_path, index=False)
    print(f"Combined feature matrix shape: {combined.shape}")
    print(f"Saved combined features to {output_path}")
    return combined


#Orchestration 
def main():
    df = load_dataset()
    distilbert_df = extract_distilbert_features(df)
    nrc_df = extract_nrc_features(df)
    combine_features(nrc_df, distilbert_df)


if __name__ == "__main__":
    main()
