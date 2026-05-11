import pickle
from app.preprocessing.preprocess import clean_text   # ✅ MUST BE HERE

MODEL_PATH = "app/models/model.pkl"
VECTORIZER_PATH = "app/models/vectorizer.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)


def predict_email(text: str):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    
    prediction = model.predict(vectorized)[0]
    probs = model.predict_proba(vectorized)[0]

    classes = model.classes_

    prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}

    return {
        "intent": prediction,
        "confidence": float(max(probs)),
        "all_probabilities": prob_dict
    }