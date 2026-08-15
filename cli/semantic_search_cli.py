import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search
from lib.search_utils import DEFAULT_SEARCH_LIMIT

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the Semantic Search Model")
    subparsers.add_parser("verify_embeddings", help="Verify the embeddings output from the Semantic Search Model")
    
    embedtext_parser = subparsers.add_parser("embed_text", help="Displays embedding info for provided text")
    embedtext_parser.add_argument("text", type=str, help="Text to embed")

    embedquery_parser = subparsers.add_parser("embed_query", help="Generates embeddings for provided query string")
    embedquery_parser.add_argument("query", type=str, help="Query to generate embedding for")

    search_parser = subparsers.add_parser("search", help="Search for similar results based on semantic search relevance")
    search_parser.add_argument("query", type=str, help="Query to compare against available documents")
    search_parser.add_argument("--limit", 
        type=int, nargs="?", 
        default=DEFAULT_SEARCH_LIMIT, 
        help="Max number of search results",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            results = search(args.query, args.limit)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"  {result['description'][:100]} ...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()