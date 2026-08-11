import os,pickle,string,math
from collections import defaultdict,Counter

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    STOPWORDS_PATH,
    BM25_K1,
    BM25_B,
    load_movies,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id, term) -> int:
        if term in self.term_frequencies[doc_id]:
            return self.term_frequencies[doc_id][term]
        return 0

    def get_idf(self, term) -> float:
        return math.log((len(self.docmap) + 1) / (len(self.get_documents(term)) + 1))

    def get_tfidf(self, doc_id, term) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)

    # log((N - df + 0.5) / (df + 0.5) + 1), where N is the total number of documents and df is the document frequency
    def get_bm25_idf(self, term) -> float:
        n = len(self.docmap)
        df = len(self.get_documents(term))
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    # Length normalization factor
    # length_norm = 1 - b + b * (doc_length / avg_doc_length)
    #
    # Apply to term frequency
    # tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_len = self.doc_lengths[doc_id]
        len_norm = (1 - b) + (b * (doc_len / self.__get_avg_doc_length()))
        return (tf * (k1 + 1)) / (tf + (k1 * len_norm))
    
    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        len_sum = 0.0
        num_docs = len(self.doc_lengths)
        for doc_id in self.doc_lengths:
            len_sum += self.doc_lengths[doc_id]
        return len_sum / num_docs

    

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movie_index = InvertedIndex()
    movie_index.load()
    if len(movie_index.index) == 0:
        print("movie index not found")
        return []
    with open('debug_index.txt', 'w') as f:
        for k,v in movie_index.index.items():
            f.write(f"k: {k}, v{v}\n")
    with open('debug_docmap.txt', 'w') as f:
        for k,v in movie_index.docmap.items():
            f.write(f"k: {k}, v{v}\n")
    results = []
    query_tokens = tokenize_text(query)
    for token in query_tokens:
        if token in movie_index.index:
            for doc_id in movie_index.get_documents(token):
                if len(results) < DEFAULT_SEARCH_LIMIT:
                    results.append(movie_index.docmap[doc_id])
                    continue
                return results
    return results

def tf_command(doc_id, term) -> int:
    movie_index = InvertedIndex()
    movie_index.load()
    token = tokenize_single_term(term)
    return movie_index.get_tf(doc_id, token)

def idf_command(term) -> float:
    movie_index = InvertedIndex()
    movie_index.load()
    token = tokenize_single_term(term)
    return movie_index.get_idf(token)

def tfidf_command(doc_id, term) -> float:
    movie_index = InvertedIndex()
    movie_index.load()
    token = tokenize_single_term(term)
    return movie_index.get_tfidf(doc_id, token)

def bm25_idf_command(term) -> float:
    movie_index = InvertedIndex()
    movie_index.load()
    token = tokenize_single_term(term)
    return movie_index.get_bm25_idf(token)

def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
    movie_index = InvertedIndex()
    movie_index.load()
    token = tokenize_single_term(term)
    return movie_index.get_bm25_tf(doc_id, token, k1=k1, b=b)

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return [preprocess_text(word) for word in f.read().splitlines()]


STOPWORDS = load_stopwords()

def tokenize_single_term(text: str) -> str:
    token = tokenize_text(text)
    if len(token) != 1:
        raise Exception("single term contains multiple tokens")
    return ''.join(token)    

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    filtered_words = []
    for word in valid_tokens:
        if word not in STOPWORDS:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words
