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
# with open(join(dirname(realpath(__file__)), "model/classifier.pkl"), "rb") as f:
#     model = joblib.load(f)

# MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "classifier.pkl")
# model = joblib.load(MODEL_PATH)

# print("Loading model from:", MODEL_PATH)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf.pkl"))
classifier = joblib.load(os.path.join(MODEL_DIR, "classifier.pkl"))

class_names = np.array([
    'Electricity Services', ' Governance and Integrity Concerns', 'Overpricing', 
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
    # complaints = np.array(data['complaints'])
    # prediction = model.predict(complaints)
    # class_name = class_names[prediction[0]]

    # complaints = data['complaints']  

    # predictions = model.predict(complaints)
    # print("After loading, is TF-IDF fitted?", hasattr(model.named_steps['tfidf'], "idf_"))

    # results = [class_names[p] for p in predictions]

    # return {'classification': results}

    complaints = data['complaints']
    if isinstance(complaints, str):
        complaints = [complaints]

    # Preprocess
    complaints_clean = preprocessor.transform(complaints)
    # Vectorize + TF-IDF
    X_counts = vectorizer.transform(complaints_clean)
    X_tfidf = tfidf.transform(X_counts)
    # Predict
    preds = classifier.predict(X_tfidf)
    results = preds.tolist() 

    return {"classification": results}