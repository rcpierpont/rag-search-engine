import argparse

from lib.keyword_search import build_command, search_command, tf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="prints term frequency for given term in given document ID")
    tf_parser.add_argument("id", type=int, help="document id")
    tf_parser.add_argument("term", type=str, help="term to search in document with given id")

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
            frequency = tf_command(args.id, args.term)
            print(f"{args.term} appears {frequency} time(s) in doc {args.id}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
