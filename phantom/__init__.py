"""
Phantom Memory — Zero-cost persistent memory for local LLMs.

Usage:
    from phantom import MemoryDaemon

    daemon = MemoryDaemon()
    daemon.start()

    daemon.ingest("user", "The cross-default threshold is $75M")
    results = daemon.recall("what's the threshold?")

    daemon.stop()
"""

from phantom.daemon import MemoryDaemon, FactExtractor, MemoryStore, VaultWriter
from phantom.enricher import PhantomEnricher

__version__ = "0.1.0"
__all__ = ["MemoryDaemon", "FactExtractor", "MemoryStore", "VaultWriter", "PhantomEnricher"]
