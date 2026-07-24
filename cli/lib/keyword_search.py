from .search_utils import load_movies, sanitize_str, remove_stopwords, stem_tokens

DEFAULT_SEARCH_LIMIT=5

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        tokens_q = preprocess(query)
        tokens_r = preprocess(movie['title'])
        if has_match(tokens_q, tokens_r):
            results.append(movie)
            if len(results) >= limit:
                break
    return results

def has_match(q: list[str], r: list[str]) -> bool:
    for token in q:
        if token in ' '.join(r):
            return True
    return False

def tokenize_str(s: str) -> list[str]:
    return s.split()

def preprocess(s: str) -> list[str]:
    sanitized_str = sanitize_str(s)
    tokens = tokenize_str(sanitized_str)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return tokens