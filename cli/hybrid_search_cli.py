import argparse
from lib.hybrid_search import normalize_scores, weighted_search_cli, rrf_search_cli
from lib.search_utils import DEFAULT_ALPHA, DEFAULT_SEARCH_LIMIT, RRF_K

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize different scales of scores to a 0-1 range so they can be compared")
    normalize_parser.add_argument("scores", type=float, nargs="*")

    weightedsearch_parser = subparsers.add_parser("weighted-search", help="Hybrid search with an alpha value to configure weight")
    weightedsearch_parser.add_argument("query", type=str, help="Text to search for in documents")
    weightedsearch_parser.add_argument("--alpha", 
        type=float, nargs="?",
        default=DEFAULT_ALPHA,
        help="Configurable weight constant allowing scores to be adjusted between exact and conceptual matches"
        )
    weightedsearch_parser.add_argument("--limit", 
        type=int, nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="number of results to fetch"
        )

    rrfsearch_parser = subparsers.add_parser("rrf-search", help="hybrid search that uses reciprocal ranking instead of calculated weights")
    rrfsearch_parser.add_argument("query", help="Text to search for in documents")
    rrfsearch_parser.add_argument("-k", 
        type=int, nargs="?",
        default=RRF_K, 
        help="configurable dropoff constant for scores"
        )
    rrfsearch_parser.add_argument("--limit", 
        type=int, nargs="?", 
        default=DEFAULT_SEARCH_LIMIT, 
        help="number of results to fetch"
        )
    
    args = parser.parse_args()

    match args.command:
        case "normalize":
            if len(args.scores) == 0:
                return

            if min(args.scores) == max(args.scores):
                for _ in range(len(args.scores)):
                    print(1.0)
                return

            normalized_scores = normalize_scores(args.scores)
            for score in normalized_scores:
                    print(f"* {score:.4f}")
        case "weighted-search":
            weighted_search_cli(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_cli(args.query, args.k, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()