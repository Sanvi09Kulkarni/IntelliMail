# IntelliMail

A transformer-based email intent analysis system built using DistilBERT, FastAPI, Streamlit, and SQLite.

IntelliMail analyzes incoming emails and predicts their intent category using contextual natural language understanding instead of traditional keyword-based classification. The application is designed to understand the actual purpose of an email, such as whether it relates to a meeting, deadline, finance, support issue, promotion, personal communication, or spam.

---

## Overview

Most traditional email filtering systems rely heavily on:
- keyword matching
- rule-based filtering
- basic machine learning models

IntelliMail improves this process using transformer-based NLP techniques that understand semantic context and sentence meaning.

The project focuses on building a complete end-to-end AI application with:
- a trained NLP model
- backend APIs
- persistent storage
- an interactive user interface

---

## Features

- Multi-class email intent classification
- Transformer-based NLP pipeline using DistilBERT
- Real-time email analysis
- Confidence score visualization
- Interactive Streamlit dashboard
- FastAPI backend integration
- SQLite database for storing prediction history
- Modern dark-themed interface
- Context-aware intent detection
- Custom fine-tuned dataset

---

## Intent Categories

| Category | Description |
|---|---|
| Meeting | Discussions, calls, reviews, sync-ups |
| Deadline | Submission reminders, due dates, urgent tasks |
| Support | Technical issues, login problems, system failures |
| Finance | Payments, invoices, billing, transactions |
| Promotion | Marketing emails, discounts, offers |
| Personal | Informal or personal communication |
| Spam | Suspicious or phishing-like emails |

---

## Example Predictions

| Email | Predicted Intent |
|---|---|
| "Please submit the assignment before tonight." | Deadline |
| "The application crashes every time I login." | Support |
| "Join the client meeting tomorrow at 3 PM." | Meeting |
| "Claim your cashback rewards before midnight." | Promotion |

---

## System Architecture

```text
Streamlit Frontend
        ↓
FastAPI Backend
        ↓
DistilBERT NLP Model
        ↓
SQLite Database
```

---

## Model Information

### DistilBERT

The project uses the pretrained transformer model:

```python
distilbert-base-uncased
```

DistilBERT was selected because it provides:
- strong contextual language understanding
- lower memory usage compared to larger transformer models
- faster inference speed
- good performance on CPU-based systems

The model was fine-tuned on a custom multi-class email dataset specifically created for intent classification.

---

## Tech Stack

### AI / NLP
- Transformers
- DistilBERT
- PyTorch
- Hugging Face

### Backend
- FastAPI
- Uvicorn

### Frontend
- Streamlit

### Database
- SQLite

### Programming Language
- Python

---

## Project Structure

```text
ai-email-analyzer/
│
├── app/
│   ├── api/
│   ├── database/
│   ├── models/
│   ├── preprocessing/
│   ├── services/
│   └── main.py
│
├── frontend/
│   └── app.py
│
├── data/
│   └── emails.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### Clone Repository

```bash
git clone <your-repository-url>
cd ai-email-analyzer
```

---

### Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

#### Windows
```bash
venv\Scripts\activate
```

#### Mac/Linux
```bash
source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Model Training

Run the following command to train the DistilBERT model:

```bash
python -m app.models.bert_train
```

This process:
- tokenizes the dataset
- fine-tunes the transformer model
- saves trained model weights

---

## Running the Backend

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

API documentation will be available at:

```text
http://127.0.0.1:8000/docs
```

---

## Running the Frontend

Launch the Streamlit application:

```bash
streamlit run frontend/app.py
```

---

## Future Improvements

Potential future enhancements include:
- top-k intent predictions
- email summarization
- analytics dashboard
- Gmail integration
- cloud deployment
- authentication and user management

---

## Learning Outcomes

This project helped strengthen understanding of:
- transformer-based NLP
- dataset engineering
- model fine-tuning
- API development with FastAPI
- frontend-backend integration
- ML debugging workflows
- confidence-based predictions
- real-world AI application architecture

---

## Author

Sanvi Kulkarni

Integrated Master's in Computer Science with specialization in Data Science
