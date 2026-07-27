import pickle,os
from .keyword_search import tokenize_str
from .search_utils import load_movies, CACHE_PATH

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]]
        self.docmap: dict[id, dict]

    def __add_document(self, doc_id, text):
        tokens = tokenize_str(text)
        for i, token in enumerate(tokens):
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term):
        return sorted(self.index[term])

    def build(self):
        movies = load_movies
        for movie in movies:
            self.docmap[movie['id']] = movie
            self.__add_document(movie['id'], f"{movie['title']} {movie['description']}")

    def save(self):
        pickle.dump(self.index, os.path.join(CACHE_PATH, 'index.pkl'))
        pickle.dump(self.docmap, os.path.join(CACHE_PATH, 'cache/docmap.pkl'))

                