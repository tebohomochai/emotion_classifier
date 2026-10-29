""""
FastAPI APP that serves the emotion classifier in 'model.py'
Like all the other python files, we run it from the project root so that the model's relative paths resolve:
    python -m uvicorn api:app --reload --app-dir src

Open http://127.0.0.1:8000/health in a browser. You should see {"status":"ok"}.    
Then open http://127.0.0.1:8000/docs
Then click POST/predict. Try it out by replacing 'string' with a headline. Then click execute.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model import EmotionClassifier

#Create the app (The application object)
#This is what uvicorn runs. Every endpoint below attaches to it.
app = FastAPI(
    title = "Emotion Classifier API",
    description = "Predicts the emotion expressed by a news headline.",
    version = "0.1.0",
)

#Load the model once, when the server starts
#Loading the model here (Not inside predict below) means it is loaded once at startup, not on every request. This is because loading DistilBERT takes a while.
classifier = EmotionClassifier()

#The two pydantic classes (requests and responses)
#Describe the shape of requests and responses. Pydantic checks incoming data against these classes automatically. Bad input gets a clear 422 error before it ever reaches the model.
class PredictRequest(BaseModel):
    text: str = Field(
        ..., #The dots mean 'required'
        min_length = 1,
        max_length = 500,
        description = "The headline to classify",
    )

class PredictResponse(BaseModel):
    label: str
    probabilities: dict[str, float]

#The endpoints (get and post)
@app.get("/health")
def health():
    """A simple check that the server is up."""
    return {"status": "ok"}

@app.post("/predict", response_model = PredictResponse)
def predict(request: PredictRequest):
    """Classify one headline and return the label plus per-class probabilities."""
    #min_length = 1 lets only spaces ("  ") through, so check for that as well
    if not request.text.strip():
        raise HTTPException(status_code = 422, detail = "text must not be blank")

    return classifier.predict(request.text)



