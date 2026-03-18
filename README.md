# ◈ Phantom Memory

**Zero-cost persistent memory for local LLMs.**

Every local LLM forgets everything between sessions. Phantom fixes that with a background daemon that extracts facts, embeds them, detects contradictions, and writes a human-readable knowledge graph — all while your GPU runs undisturbed.

```bash
pip install phantom-memory
```

```python
from phantom import MemoryDaemon

daemon = MemoryDaemon()
daemon.start()

daemon.ingest("user", "Cross-default threshold is $75M for Counterparty Alpha")
results = daemon.recall("what's the threshold?")
# → [{"text": "Cross-default threshold is $75M...", "score": 0.89, ...}]

daemon.stop()
```

## How It Works

Phantom runs three tiers concurrently on Apple Silicon (CPU tiers work on any hardware):

| Tier | Hardware | What It Does | Speed |
|------|----------|-------------|-------|
| **Extraction** | CPU | Regex-based fact extraction, type classification, noise filtering | Instant |
| **Embedding** | CPU | sentence-transformers (22M params) → ChromaDB vector store | 1,721 emb/sec |
| **Classification** | ANE *(optional)* | CoreML fact classifier on Neural Engine | Zero GPU cost |

**Your GPU is never touched.** The daemon runs in a background thread. We measured -4.2% GPU throughput impact — within noise.

### What Gets Extracted

Every conversation turn is parsed for:
- **Entities** — companies, people, financial instruments, legal provisions
- **Quantities** — dollar amounts, percentages, time periods
- **Decisions** — "we decided to...", "going with...", "approved..."
- **Tasks** — "need to...", "deadline is...", "follow up on..."
- **Preferences** — "we always use...", "our policy is..."

Noise is filtered automatically. "Sure, sounds good" and "I can help with that" don't pollute your memory.

### Deduplication & Contradiction Detection

- **Dedup**: Facts above 95% cosine similarity are skipped
- **Contradictions**: When a new fact about the same entities has different quantities (e.g., "threshold is $50M" → "threshold increased to $75M"), the old fact is marked as **superseded**
- **Temporal decay**: 7-day half-life — recent facts score higher than old ones

### Obsidian-Compatible Knowledge Graph

Every fact is written to structured markdown files:

```
vault/
  memory/
    entities/
      Counterparty-Alpha.md    # All facts about this entity
      ISDA-Master-Agreement.md
    decisions/
      decisions.md             # All decisions, chronological
    tasks/
      tasks.md                 # Open tasks and deadlines
    preferences/
      preferences.md           # User/team preferences
    sessions/
      2026-03-17-1430.md       # Per-session summaries
```

Entity files use Obsidian `[[wikilinks]]` for cross-referencing. Superseded facts get ~~strikethrough~~ with timestamps.

## Quick Start

### With Ollama

```python
from phantom import MemoryDaemon

daemon = MemoryDaemon(vault_path="~/obsidian/vault")
daemon.start()

# Your existing Ollama chat loop
response = requests.post("http://localhost:11434/api/chat", json={...})

# Add these two lines to your loop:
daemon.ingest("user", user_message)
daemon.ingest("assistant", response_text)

# Before each request, inject memories:
context = daemon.recall_formatted(user_message)
# → "## Relevant Memories\n- [decision] Threshold set at $75M (score=0.89)"
```

See [`examples/`](examples/) for complete working examples with Ollama, MLX, and any OpenAI-compatible server.

### CLI

```bash
# Start daemon + web dashboard
phantom start --vault ~/obsidian/vault

# Search memories
phantom recall "cross-default threshold"

# Check stats
phantom stats

# Manually ingest a fact
phantom ingest "Meeting with John at 3pm tomorrow"
```

### Dashboard

The web dashboard at `http://localhost:8422` shows:
- **Knowledge Graph** — all entities with fact counts
- **Activity Feed** — real-time fact extraction with type color coding
- **System Stats** — memory counts, dedup rates, contradiction tracking
- **Decisions & Tasks** — filtered views of actionable items

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Your LLM (Ollama / MLX / LM Studio / vLLM)    │  ← GPU, undisturbed
└──────────────────────┬──────────────────────────┘
                       │ conversation turns
                       ▼
┌─────────────────────────────────────────────────┐
│  Phantom Daemon (background thread)             │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Extractor │→│ Embedder │→│ ChromaDB     │  │  ← CPU only
│  │ (regex)   │  │ (22M)    │  │ (vector DB)  │  │
│  └──────────┘  └──────────┘  └──────┬───────┘  │
│                                      │          │
│  ┌──────────────────────────────────┐│          │
│  │ Vault Writer (Obsidian markdown) ││          │  ← disk I/O
│  └──────────────────────────────────┘│          │
│                                      │          │
│  ┌──────────────────────────────────┐│          │
│  │ Contradiction Detector           ││          │  ← CPU
│  │ (supersedes old facts)           ││          │
│  └──────────────────────────────────┘│          │
└─────────────────────────────────────────────────┘
```

## Benchmarks

Tested on MacBook Air M5 (16GB) running Qwen 3.5 9B via MLX:

| Metric | Value |
|--------|-------|
| Embedding speed | 1,721 embeddings/sec (CPU) |
| GPU impact during daemon operation | -4.2% (within noise) |
| Fact extraction | Instant (regex, no model) |
| Retrieval accuracy at 526 facts | 88.9% |
| Contradiction detection | 0% leaked (40% → 0% with supersession) |
| Memory overhead | ~180 MB (embedding model + ChromaDB) |

## Apple Silicon Bonus

On Apple Silicon, Phantom can optionally use the **Neural Engine** for fact classification at zero GPU cost. Enable with:

```python
daemon = MemoryDaemon(vault_path="~/vault")
# ANE classifier at port 8423 (when available)
```

The ANE tier is completely optional. Everything works on CPU alone — Linux, Windows, Intel Macs, whatever. Apple Silicon users just get an extra gear.

## API Reference

### `MemoryDaemon`

```python
daemon = MemoryDaemon(
    vault_path="~/vault",           # Obsidian vault path (default: ./phantom_vault)
    db_path="~/phantom_db",         # ChromaDB path (default: auto)
    session_id="my-session",        # Session identifier (default: timestamp)
    embedding_model="all-MiniLM-L6-v2"  # Any sentence-transformers model
)

daemon.start()                      # Start background processing
daemon.ingest("user", "text")       # Feed a conversation turn (non-blocking)
daemon.recall("query", n_results=5) # Semantic search with temporal decay
daemon.recall_formatted("query")    # Pre-formatted for LLM context injection
daemon.stats                        # {"stored": 42, "deduped": 3, ...}
daemon.stop()                       # Stop and write session summary
```

### `MemoryStore` (direct access)

```python
from phantom import MemoryStore

store = MemoryStore(db_path="~/phantom_db")
store.recall("what's the threshold?", n_results=5)
store.count()  # total facts stored
```

### `FactExtractor` (standalone)

```python
from phantom import FactExtractor

extractor = FactExtractor()
facts = extractor.extract("The threshold is $75M for Counterparty Alpha")
# [{"text": "...", "type": "quantitative", "entities": ["Counterparty Alpha"], "quantities": ["$75M"]}]
```

## License

MIT
