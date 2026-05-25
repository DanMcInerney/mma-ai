"""Command line entry point for the MMA AI web dashboard."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("MMA_AI_HOST", "0.0.0.0")
    port = int(os.getenv("MMA_AI_PORT", "8000"))
    uvicorn.run("libs.web.app:app", host=host, port=port, reload=os.getenv("MMA_AI_RELOAD") == "1")


if __name__ == "__main__":
    main()
