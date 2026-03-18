#!/usr/bin/env python3
"""
Phantom CLI — Command-line interface for the memory daemon.

Usage:
    phantom start                    # Start daemon + dashboard
    phantom start --no-dashboard     # Daemon only
    phantom recall "search query"    # Search memories
    phantom stats                    # Show memory stats
    phantom ingest "some text"       # Manually ingest a fact
"""

import argparse
import sys
import time
import signal
import os

def cmd_start(args):
    """Start the memory daemon (and optionally the dashboard)."""
    from phantom.daemon import MemoryDaemon

    vault_path = os.path.expanduser(args.vault or "~/phantom_vault")
    db_path = os.path.expanduser(args.db) if args.db else None

    daemon = MemoryDaemon(vault_path=vault_path, db_path=db_path)
    daemon.start()

    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  ◈ PHANTOM MEMORY                                ║")
    print(f"║  Session: {daemon.session_id:38s}  ║")
    print(f"║  Vault:   {vault_path[:38]:38s}  ║")
    print(f"║  Memories: {daemon.store.count():<37d}  ║")
    print(f"╚══════════════════════════════════════════════════╝")

    if not args.no_dashboard:
        print(f"\n  Dashboard: http://localhost:{args.port}")
        try:
            from phantom.dashboard import create_app
            from aiohttp import web
            app = create_app(vault_path=vault_path, db_path=db_path or os.path.join(os.path.dirname(vault_path), "phantom_db"))
            web.run_app(app, host='localhost', port=args.port, print=lambda *a: None)
        except ImportError:
            print("  (aiohttp not available — running daemon only)")
            _wait_forever(daemon)
    else:
        print("\n  Running in daemon-only mode. Ctrl+C to stop.")
        _wait_forever(daemon)


def _wait_forever(daemon):
    """Block until Ctrl+C, then stop daemon."""
    def _handle_signal(sig, frame):
        print("\n  Stopping daemon...")
        daemon.stop()
        stats = daemon.stats
        print(f"  Final stats: {stats['stored']} facts stored, {stats['deduped']} deduped, {stats.get('superseded', 0)} superseded")
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    while True:
        time.sleep(1)


def cmd_recall(args):
    """Search memories."""
    from phantom.daemon import MemoryStore

    db_path = os.path.expanduser(args.db) if args.db else os.path.expanduser("~/phantom_db")
    if not os.path.exists(db_path):
        print(f"No memory store found at {db_path}")
        print("Run 'phantom start' first to create one, or specify --db PATH")
        sys.exit(1)

    store = MemoryStore(db_path)
    results = store.recall(args.query, n_results=args.n)

    if not results:
        print("No matching memories found.")
        return

    print(f"\n  Found {len(results)} memories for: \"{args.query}\"\n")
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        superseded = " [SUPERSEDED]" if r["superseded"] else ""
        print(f"  {i}. [{meta.get('type', '?')}] {r['text']}")
        print(f"     score={r['score']:.3f}  similarity={r['similarity']:.3f}  recency={r['recency']:.3f}{superseded}")
        print()


def cmd_stats(args):
    """Show memory store stats."""
    from phantom.daemon import MemoryStore

    db_path = os.path.expanduser(args.db) if args.db else os.path.expanduser("~/phantom_db")
    if not os.path.exists(db_path):
        print(f"No memory store found at {db_path}")
        sys.exit(1)

    store = MemoryStore(db_path)
    count = store.count()

    print(f"\n  ◈ Phantom Memory Store")
    print(f"  Path: {db_path}")
    print(f"  Total memories: {count}")
    print()


def cmd_ingest(args):
    """Manually ingest a fact."""
    from phantom.daemon import MemoryDaemon

    vault_path = os.path.expanduser(args.vault or "~/phantom_vault")
    db_path = os.path.expanduser(args.db) if args.db else None

    daemon = MemoryDaemon(vault_path=vault_path, db_path=db_path)
    daemon.start()
    daemon.ingest(args.role, args.text)
    time.sleep(1)  # Let background thread process
    daemon.stop()

    stats = daemon.stats
    print(f"  Ingested: {stats['extracted']} facts extracted, {stats['stored']} stored")


def main():
    parser = argparse.ArgumentParser(
        prog="phantom",
        description="Phantom Memory — Zero-cost persistent memory for local LLMs",
    )
    parser.add_argument("--vault", type=str, default=None, help="Path to Obsidian vault (default: ~/phantom_vault)")
    parser.add_argument("--db", type=str, default=None, help="Path to ChromaDB storage (default: ~/phantom_db)")

    subparsers = parser.add_subparsers(dest="command")

    # start
    start_parser = subparsers.add_parser("start", help="Start the memory daemon")
    start_parser.add_argument("--no-dashboard", action="store_true", help="Don't start the web dashboard")
    start_parser.add_argument("--port", type=int, default=8422, help="Dashboard port (default: 8422)")

    # recall
    recall_parser = subparsers.add_parser("recall", help="Search memories")
    recall_parser.add_argument("query", type=str, help="Search query")
    recall_parser.add_argument("-n", type=int, default=5, help="Number of results")

    # stats
    subparsers.add_parser("stats", help="Show memory store statistics")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Manually ingest a fact")
    ingest_parser.add_argument("text", type=str, help="Text to ingest")
    ingest_parser.add_argument("--role", type=str, default="user", choices=["user", "assistant"], help="Source role")

    args = parser.parse_args()

    if args.command == "start":
        cmd_start(args)
    elif args.command == "recall":
        cmd_recall(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
