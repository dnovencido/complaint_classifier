from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from nltk.corpus import stopwords
import numpy as np
import os
import joblib
from app.preprocess import TextPreprocessor
from os.path import join, dirname, realpath

# Load the saved model
# model = joblib.load("app/model/complaint_classifier.pkl")
with open(join(dirname(realpath(__file__)), "model/complaint_classifier.pkl"), "rb") as f:
    model = joblib.load(f)

class_names = np.array([
    'Electricity Services', 'Governance and Integrity Concerns', 'Overpricing', 
    'Procedural and Operational Issues', 'Roads & Infrastructure', 'Service Delays and Inaction'
    'Transportation', 'Unprofessional Conduct and Behavior', 'Waste Management', 'Water Services', 'Workforce Concerns'
])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],
)

@app.get('/')
def read_root():
    return {'message': 'Model API'}

@app.post('/classify')

def predict(data: dict):
    """
    Predicts the class of a given set of features.

    Args:
        data (dict): A dictionary containing the features to predict.
        e.g. {"complaints": ["complaint 1", "complaint 2", ...]}

    Returns:
        dict: A dictionary containing the predicted class.
    """        

    complaints = data['complaints']

    if isinstance(complaints, str):
        complaints = [complaints]

    predictions = model.predict(complaints)
    results = predictions.tolist() 

    return {"classification": results}