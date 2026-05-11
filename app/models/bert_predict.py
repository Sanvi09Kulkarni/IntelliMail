import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)


# =========================
# LOAD MODEL + TOKENIZER
# =========================

MODEL_PATH = "app/models/bert_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)

model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()


# =========================
# LOAD LABEL MAPS
# =========================

label2id = torch.load(f"{MODEL_PATH}/label2id.pt")

id2label = torch.load(f"{MODEL_PATH}/id2label.pt")


# =========================
# PREDICTION FUNCTION
# =========================

def predict_email_bert(text: str):

    # tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    # model prediction
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    # probabilities
    probs = torch.softmax(logits, dim=1)

    predicted_class_id = torch.argmax(probs, dim=1).item()

    confidence = probs[0][predicted_class_id].item()

    predicted_label = id2label[predicted_class_id]

    return {
        "intent": predicted_label,
        "confidence": round(confidence, 4)
    }