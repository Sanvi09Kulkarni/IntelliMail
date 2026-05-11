import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    
    return " ".join(words)