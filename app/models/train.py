import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from app.preprocessing.preprocess import clean_text

MODEL_PATH = "app/models/model.pkl"
VECTORIZER_PATH = "app/models/vectorizer.pkl"


def train_model():
    df = pd.read_csv("data/emails.csv")

    # Preprocess
    df["clean_text"] = df["text"].apply(clean_text)

    X = df["clean_text"]
    y = df["label"]

    # 🔥 Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)

    # 🔥 Evaluation
    y_pred = model.predict(X_test_vec)

    print("\n📊 Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print("\n✅ Model trained and saved!")


if __name__ == "__main__":
    train_model()