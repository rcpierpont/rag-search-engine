import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, semantic_search, chunk_text, semantic_chunk_text, embed_chunks, search_chunked
from lib.search_utils import DEFAULT_SEARCH_LIMIT, DEFAULT_CHUNK_SIZE, DEFAULT_SEMANTIC_CHUNK_SIZE

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

    chunk_parser = subparsers.add_parser("chunk", help="Split text into smaller chunks with optional overlap")
    chunk_parser.add_argument("text", type=str, help="Text that will be chunked")
    chunk_parser.add_argument("--chunk-size",
        type=int, nargs="?",
        default=DEFAULT_CHUNK_SIZE,
        help="String length maximum for each chunk",
        )
    chunk_parser.add_argument("--overlap", 
        type=int, nargs="?",
        default=0,
        help="How many words to overlap between chunks (helps to retain original context)",
        )

    semanticchunk_parser = subparsers.add_parser("semantic_chunk", help="Splits text into chunks, keeping sentence boundaries to preserve more meaning")
    semanticchunk_parser.add_argument("text", type=str, help="Text that will be chunked")
    semanticchunk_parser.add_argument("--max-chunk-size",
        type=int, nargs="?",
        default=DEFAULT_SEMANTIC_CHUNK_SIZE,
        help="String length maximum for each chunk",
        )
    semanticchunk_parser.add_argument("--overlap", 
        type=int, nargs="?",
        default=0,
        help="How many words to overlap between chunks (helps to retain original context)",
        )

    subparsers.add_parser("embed_chunks", help="Generate vectorized embeddings for chunked text")

    searchchunked_parser = subparsers.add_parser("search_chunked", help="Searches by comparing embeddings of query chunks against saved embeddings")
    searchchunked_parser.add_argument("query", type=str, help="Text used in search")
    searchchunked_parser.add_argument("--limit", 
        type=int, nargs="?",
        default=5,
        help="Max size of search results sorted by comparison score"
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
            semantic_search(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            search_chunked(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()