import argparse
from lib.keyword_search import search_command
from lib.build import build_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
            pass
        case "build":
            doc_index = build_command()
            docs = doc_index.index['merida']
            print(f"First document for token 'merida' = {docs[0]}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()