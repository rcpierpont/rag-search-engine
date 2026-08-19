import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import SearchResult, Movie, load_movies
from collections import defaultdict
from typing import TypedDict

class HybridSearchResult(TypedDict):
    id: int
    title: str
    document: str
    scores: dict[str, float]

class RRFSearchResult(TypedDict):
    id: int
    title: str
    document: str
    bm25_rank: int
    semantic_rank: int
    rrf_score: float

class HybridSearch:
    def __init__(self, documents: list[Movie]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[SearchResult]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        bm25_scores = normalize_scores([result['score'] for result in bm25_results])

        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        sem_scores = normalize_scores([result['score'] for result in semantic_results])

        doc_scores: defaultdict[int, dict] = defaultdict()
        for i, result in enumerate(bm25_results):
            if doc_scores.get(result['id']) is None:
                doc_scores[result['id']] = defaultdict(dict[str, float])
            doc_scores[result['id']]['keyword_score'] = bm25_scores[i]
        for j, result in enumerate(semantic_results):
            if doc_scores.get(result['id']) is None:
                doc_scores[result['id']] = defaultdict(dict[str, float])
            doc_scores[result['id']]['semantic_score'] = sem_scores[j]

        for doc_id in doc_scores:
            doc_scores[doc_id]['hybrid_score'] = hybrid_score(
                doc_scores[doc_id]['keyword_score'], 
                doc_scores[doc_id]['semantic_score'],
                alpha
            )
        
        sorted_doc_scores = sorted(doc_scores.items(), key=lambda item: item[1]['hybrid_score'], reverse=True)
        results = []
        for doc_id, doc_scores in sorted_doc_scores:
            results.append(HybridSearchResult({
                'id': doc_id,
                'title': self.idx.docmap[doc_id]['title'],
                'document': self.idx.docmap[doc_id]['description'],
                'scores': doc_scores,
            }))
        return results

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        bm25_sorted = sorted(bm25_results, key=lambda x: x['score'], reverse=True)

        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        semantic_sorted = sorted(semantic_results, key=lambda x: x['score'], reverse=True)

        doc_ranks: defaultdict[int, dict] = defaultdict()
        for i, result in enumerate(bm25_sorted, 1):
            if doc_ranks.get(result['id']) is None:
                doc_ranks[result['id']] = {
                    'title': result['title'],
                    'document': result['document'],
                    'bm25_rank': i
                }
                continue
            doc_ranks[result['id']]['bm25_rank'] = i

        for j, result in enumerate(semantic_sorted):
            if doc_ranks.get(result['id']) is None:
                doc_ranks[result['id']] = {
                    'title': result['title'],
                    'document': result['document'],
                    'semantic_rank': j
                }
                continue
            doc_ranks[result['id']]['semantic_rank'] = j

        for doc_id in doc_ranks:
            bm25_rrf, semantic_rrf = 0,0
            if doc_ranks[doc_id].get('bm25_rank') is not None:
                bm25_rrf = rrf_score(doc_ranks[doc_id]['bm25_rank'], k)
            if doc_ranks[doc_id].get('semantic_rank') is not None:
                semantic_rrf = rrf_score(doc_ranks[doc_id]['semantic_rank'], k)
            doc_ranks[doc_id]['rrf_score'] = bm25_rrf + semantic_rrf
        rrf_sorted = sorted(doc_ranks.items(), key=lambda item: item[1]['rrf_score'], reverse=True)

        results = []
        for doc_id, doc_data in rrf_sorted:
            results.append(RRFSearchResult({
                'id': doc_id,
                'title': doc_data['title'],
                'document': doc_data['document'],
                'bm25_rank': doc_data['bm25_rank'],
                'semantic_rank': doc_data['semantic_rank'],
                'rrf_score': doc_data['rrf_score'],
            }))

        return results

def normalize_scores(scores: list[float]) -> list[float]:
    score_min, score_max = min(scores), max(scores)
    normalized_scores = []
    for score in scores:
        normalized_scores.append((score - score_min) / (score_max - score_min))
    return normalized_scores

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)

def weighted_search_cli(query: str, alpha: float, limit: int = 5):
    documents = load_movies()
    search_instance = HybridSearch(documents)
    results = search_instance.weighted_search(query, alpha, limit)
    for i, result in enumerate(results[:limit], 1):
        print(f"{i}. {result['title']}")
        print(f"   Hybrid Score: {result['scores']['hybrid_score']:.4f}")
        print(f"   BM25: {result['scores']['keyword_score']:.4f}, Semantic: {result['scores']['semantic_score']:.4f}")
        print(f"   {result['document'][:100]}...")

def rrf_search_cli(query: str, k: int, limit: int = 5):
    documents = load_movies()
    search_instance = HybridSearch(documents)
    results = search_instance.rrf_search(query, k, limit)
    for i, result in enumerate(results[:limit], 1):
        print(f"{i}. {result['title']}")
        print(f"   RRF Score: {result['rrf_score']:.3f}")
        print(f"   BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}")
        print(f"   {result['document']}...")
