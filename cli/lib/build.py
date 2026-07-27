from .inverted_index import InvertedIndex

def build_command() -> InvertedIndex:
    index = InvertedIndex()
    index.build()
    index.save()
    return index