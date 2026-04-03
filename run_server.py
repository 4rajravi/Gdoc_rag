#!/usr/bin/env python3
"""
Start the FastAPI backend server.

Usage:
    python run_server.py
    python run_server.py --port 8000 --reload
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start the RAG API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()