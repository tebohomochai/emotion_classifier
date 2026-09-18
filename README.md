# Emotion classifier: XGBoost using DistilBERT+NRC features

Categorical emotion classification of news headlines, using a hybrid feature
set of contextual embeddings from DistilBERT combined with lexicon-based
affect frequencies from the NRC Emotion Lexicon. Adapted from an MCom thesis on emotion classification, and restructured
here as a runnable project rather than a notebook.

## Problem

Given a news headline, predict which of the following six emotion classes it conveys:
`anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise`.

## Approach

* **Dataset**: SemEval-2007 headlines, balanced to 2700 instances (450 per
class). The dataset was augmented using the Conditional-BERT model.
* **Features**:

  * *DistilBERT*: `\[CLS]` token embedding from `distilbert-base-uncased`'s
last hidden state of 768 dimensions.
  * *NRC*: proportion-based affect frequencies across 8 emotion categories and 2 sentiment
 polarities from `NRCLex`.
  * Combined into a single feature matrix with 778 dimensions.
* **Model**: XGBoost, tuned via `RandomizedSearchCV` over 30 iterations, using 5-fold
cross-validation.

## Project structure

```
emotion_classifier/
├── src/
│   ├── preprocess.py   #feature extraction (NRC+DistilBERT), cached to disk
│   ├── train.py        #hyperparameter search, training and final evaluation
│   ├── evaluate.py      #classification report, confusion matrix and ROC curves
│   └── model.py         #prediction wrapper for a single headline
├── data/                 #raw dataset + cached feature files (gitignored)
├── models/               #trained model + label encoder (gitignored)
├── results/              #exported ROC data (gitignored)
└── notebooks/            #original exploratory notebook, kept for reference
```

## Setup

```bash
pip install -r requirements.txt
```

Place the raw dataset at `data/final\_augmented\_dataset.csv` (semicolon-delimited,
`text` and `label` columns).

## Running the pipeline

```bash
#Extract and cache features (skips steps if cached files already exist)
python src/preprocess.py

#Train, tune and evaluate the model
python src/train.py
```

## Predicting on new text

```python
from model import EmotionClassifier

clf = EmotionClassifier()
result = clf.predict("Stocks plunge after surprise rate hike")
print(result\["label"], result\["probabilities"])
```

## Results summary

Across classifiers and feature sets evaluated in the thesis:

* The four negative emotion classes (anger, disgust, fear, sadness) share
near-identical sentiment polarity profiles, making them consistently the
hardest to distinguish from one another.
* `joy` and `disgust` alternate as the best-classified class depending on
the feature set used.
* `surprise` is consistently the weakest-performing class across all
classifiers and feature sets.

See `results/roc\_xgb\_distilbert\_nrc.json` for this model's class-specific ROC data
after running `train.py`.

