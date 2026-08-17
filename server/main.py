import argparse
from waitress import serve

from app import app


def parse_args():
    parser = argparse.ArgumentParser(description="Neural chess bot HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    serve(app, host=args.host, port=args.port, threads=8)
    