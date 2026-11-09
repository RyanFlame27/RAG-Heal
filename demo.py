"""CLI entry point for the same final pipeline used by the web dashboard."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', default='In what year did the student paper Common Sense begin publication at Notre Dame?')
    args = parser.parse_args()
    from src.api_server import initialize_runtime, process_query_endpoint, QueryRequest
    initialize_runtime()
    result = process_query_endpoint(QueryRequest(query=args.query))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
