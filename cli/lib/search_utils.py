import json, os, string
from nltk.stem import PorterStemmer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

def remove_symbols(s: str) -> str:
    symbol_trans = str.maketrans({c: '' for c in string.punctuation})
    return s.translate(symbol_trans)

def sanitize_str(s: str) -> str:
    sanitized_s = s.lower()
    sanitized_s = remove_symbols(sanitized_s)
    return sanitized_s

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return [sanitize_str(word) for word in f.read().splitlines()]

STOPWORDS = load_stopwords()

def load_movies() -> list:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def remove_stopwords(tokens: list[str]) -> list[str]:
    tokens_cleaned = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        tokens_cleaned.append(token)
    return tokens_cleaned

def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    for i, token in enumerate(tokens):
        tokens[i] = stemmer.stem(token)
    return tokens
