import numpy as np
from numpy.typing import NDArray
import os, re, json
from typing import Any, TypedDict
from sentence_transformers import SentenceTransformer
from .search_utils import (
    MOVIE_EMBEDDINGS_PATH, CHUNK_EMBEDDINGS_PATH, CHUNK_METADATA_PATH,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_SEMANTIC_CHUNK_SIZE,
    Movie, load_movies, format_search_result
    )

class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int

class SemanticSearchResult(TypedDict):
    score: float
    title: str
    description: str

EmbeddingArray = NDArray[Any]

class SemanticSearch:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings: EmbeddingArray | None = None
        self.documents: list[Movie] | None = None
        self.document_map: dict[int, Movie] = {}

    def build_embeddings(self, documents: list[Movie]):
        self.documents = documents
        self.document_map = {}
        movie_strings: list[str] = []
        for doc in self.documents:
            self.document_map[doc['id']] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)

        os.makedirs(os.path.dirname(MOVIE_EMBEDDINGS_PATH), exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]):
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc['id']] = doc

        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("text parameter cannot be empty")
        return self.model.encode([text])[0]

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        scores: list[tuple] = []
        for i, doc_embedding in enumerate(self.embeddings, 0):
            score = cosine_similarity(query_embedding, doc_embedding)
            scores.append((score, self.documents[i]))
        sorted_by_score = sorted(scores, key=lambda item: item[0], reverse=True)
        results = []
        for score, movie in sorted_by_score[:limit]:
            results.append(SemanticSearchResult(
                score=score,
                title=movie['title'],
                description=movie['description'],
            ))
        return results

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata: list[ChunkMetadata] = []

    def build_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc['id']] = doc

        movie_chunks: list[str] = []
        metadata: list[ChunkMetadata] = []

        for i, doc in enumerate(self.documents):
            if len(doc['description']) == 0:
                continue
            desc_chunks = semantic_chunking(doc['description'], max_chunk_size=4, overlap=1)
            n = len(desc_chunks)
            for k, chunk in enumerate(desc_chunks):
                movie_chunks.append(chunk)
                metadata.append({
                    'movie_idx': i,
                    'chunk_idx': k,
                    'total_chunks': n,
                })
        self.chunk_embeddings = self.model.encode(movie_chunks, show_progress_bar=True, convert_to_numpy=True)
        self.chunk_metadata = metadata
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, 'w') as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(movie_chunks)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc['id']] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, 'r') as f:
                self.chunk_metadata = json.load(f)

            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        q_embedding = self.generate_embedding(query)
        chunk_scores = []
        for i, embedding in enumerate(self.chunk_embeddings):
            chunk_scores.append({
                'chunk_idx': self.chunk_metadata['chunks'][i]['chunk_idx'],
                'movie_idx': self.chunk_metadata['chunks'][i]['movie_idx'],
                'overall_chunk_idx': i,
                'score': cosine_similarity(q_embedding, embedding),
            })
        sorted_by_score = sorted(chunk_scores, key=lambda item: item['score'], reverse=True)
        score_dict = {}
        for chunk in sorted_by_score:
            if score_dict.get(chunk['movie_idx']) is None or score_dict[chunk['movie_idx']]['score'] < chunk['score']:
                score_dict[chunk['movie_idx']] = {
                    'score': chunk['score'],
                    'metadata': self.chunk_metadata['chunks'][chunk['overall_chunk_idx']],
                }
        #score_dict = {chunk['movie_idx']: chunk['score'] for chunk in sorted_by_score[0 : limit]}
        results = []
        for movie_idx, meta_dict in score_dict.items():
            if len(results) == limit:
                break
            results.append(format_search_result(
                doc_id=self.documents[movie_idx]['id'],
                title=self.documents[movie_idx]['title'],
                document=self.documents[movie_idx]['description'][:100],
                score=round(meta_dict['score'], 4),
                metadata=meta_dict['metadata']
            ))
        print(f'num results: {len(results)}')
        return results

def verify_model():
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")

def verify_embeddings():
    search_instance = SemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )

def embed_text(text):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def embed_query_text(query):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_search(query, limit) -> list[SemanticSearchResult]:
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)

    results = search_instance.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()

def fixed_size_chunking(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words:
        chunk_words = words[i : i + chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break

        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap

    return chunks

def semantic_chunking(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []

    n_sentences = len(sentences)
    i = 0
    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break

        chunks.append(" ".join(chunk_sentences))
        i += max_chunk_size - overlap

    return chunks
    
def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")

def semantic_chunk_text(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = semantic_chunking(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")

def embed_chunks() -> None:
    search_instance = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")

def search_chunked(query: str, limit: int):
    search_instance = ChunkedSemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_chunk_embeddings(documents)
    results = search_instance.search_chunks(query, limit)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")
    