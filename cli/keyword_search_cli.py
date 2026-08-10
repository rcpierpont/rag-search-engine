import argparse
from lib.keyword_search import build_command, search_command, tf_command, idf_command, tfidf_command, bm25_idf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get TF (term frequency) for given term and document ID")
    tf_parser.add_argument("doc_id", type=int, help="document id")
    tf_parser.add_argument("term", type=str, help="term to search in document with given id")

    idf_parser = subparsers.add_parser("idf", help="Get IDF score for given term")
    idf_parser.add_argument("term", type=str, help="term to search for across all documents")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TFIDF score for given term and document ID")
    tfidf_parser.add_argument("doc_id", type=int, help="document id")
    tfidf_parser.add_argument("term", type=str, help="term to search in document with given id")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM24 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building index...")
            build_command()
            print("Index built successfully.")
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, result in enumerate(results, 1):
                print(f"{i}. ({result['id']}) {result['title']}")
        case "tf":
            print("Checking term frequency...")
            frequency = tf_command(args.doc_id, args.term)
            print(f"{args.term} appears {frequency} time(s) in doc {args.doc_id}")
        case "idf":
            print("Checking idf score...")
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            print("Checking TF-IDF score...")
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            print("Checking BM25 IDF score...")
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
