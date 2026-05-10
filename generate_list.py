#!/usr/bin/env python3
"""
CLI to generate a recommendation list without the MCP server.

Usage:
    python generate_list.py [--mode strict|relaxed] [--notes "..."]
"""
import argparse
import logging
import socket
import sys
from pathlib import Path

# Make sure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

# Prevent engine network calls from hanging indefinitely
socket.setdefaulttimeout(15)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Generate a music recommendation list")
    parser.add_argument("--mode", choices=["strict", "relaxed"], default="strict")
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()

    print("Initializing system…")
    from agent_tools.tools import start_session

    result = start_session(diversity_mode=args.mode, notes=args.notes)

    session_id = result["session_id"]
    suggestions = result.get("suggestions", [])
    allocation = result.get("engine_allocation", {})
    playlist_url = result.get("candidate_playlist_url", "N/A")

    print(f"\nSession ID : {session_id}")
    print(f"Playlist   : {playlist_url}")
    print(f"Engine slots: {allocation}")
    print(f"\n{'#':>2}  {'Title':<40} {'Artist':<30} {'Genre':<20} Engine")
    print("-" * 110)
    for i, s in enumerate(suggestions, 1):
        title = s.get("title", "?")[:39]
        artist = s.get("artist", "?")[:29]
        genre = s.get("genre", "?")[:19]
        engine = s.get("engine", "?")
        print(f"{i:>2}  {title:<40} {artist:<30} {genre:<20} {engine}")

    if not suggestions:
        print("(no suggestions — check logs for engine errors)")


if __name__ == "__main__":
    main()
