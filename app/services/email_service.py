from app.models.bert_predict import predict_email_bert
from app.database.db import get_connection

def process_email(text: str):
    # Step 1: Predict
    result = predict_email_bert(text)

    # Step 2: Store in DB
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO emails (text, prediction, confidence)
        VALUES (?, ?, ?)
    """, (text, result["intent"], result["confidence"]))

    conn.commit()
    conn.close()

    return result
def fetch_emails(label=None, limit=10):
    conn = get_connection()
    cursor = conn.cursor()

    if label:
        cursor.execute(
            "SELECT * FROM emails WHERE prediction = ? ORDER BY timestamp DESC LIMIT ?",
            (label, limit)
        )
    else:
        cursor.execute(
            "SELECT * FROM emails ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": row[0],
            "text": row[1],
            "prediction": row[2],
            "confidence": row[3],
            "timestamp": row[4]
        }
        for row in rows
    ]