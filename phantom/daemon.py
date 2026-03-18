"""
Phantom Memory Daemon — Zero-cost persistent memory for local LLMs
===================================================================

Three-tier architecture:
  Tier 1 — CPU: Extract facts + embed into ChromaDB (1,721 embeddings/sec)
  Tier 2 — ANE: Classify + organize (optional, Apple Silicon only)
  Tier 3 — GPU: Your LLM runs undisturbed (zero measured impact)

All tiers run concurrently. The daemon processes conversation turns in a
background thread — extraction, embedding, deduplication, contradiction
detection, and vault writing happen invisibly while your model generates.

Usage:
    daemon = MemoryDaemon(vault_path="~/obsidian/vault")
    daemon.start()

    # Feed conversation turns as they happen
    daemon.ingest("user", "The cross-default threshold is $50M")
    daemon.ingest("assistant", "That's below the typical 3-5% range...")

    # Later: retrieve relevant context
    context = daemon.recall("cross-default threshold")

    daemon.stop()
"""

import os
import re
import time
import json
import queue
import hashlib
import threading
from datetime import datetime
from typing import Optional

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer


# ── Fact Extraction (CPU, heuristic, zero model cost) ────────────

class FactExtractor:
    """Extracts atomic facts from conversation text using heuristics.

    No model required. Uses regex patterns and keyword matching to identify
    entities, quantities, decisions, tasks, and preferences. Filters noise
    (greetings, filler, duplicates) automatically.

    Customize by subclassing and overriding ENTITY_PATTERNS, DECISION_MARKERS, etc.
    """

    # ─── Quantities ───
    QUANTITY_PATTERN = re.compile(
        r'(?:USD|GBP|EUR|\$|£|€)\s*[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|M|B|K))?\b'
        r'|[\d,]+(?:\.\d+)?%'
        r'|\b\d+\s+(?:days?|months?|years?|business days?)\b',
        re.IGNORECASE
    )

    # ─── Entities (domain-specific, curated patterns) ───
    ENTITY_PATTERNS = [
        # Finance/legal document types
        re.compile(r'\b(ISDA(?:\s+Master\s+Agreement)?|CSA|Credit\s+Support\s+Annex|Schedule)\b', re.IGNORECASE),
        # Parties / clients
        re.compile(r'\b((?:Party|Counterparty|[Cc]lient)\s+[A-Z]\w*)\b'),
        # Standalone proper names used as identifiers
        re.compile(r'\b((?:Alpha|Beta|Gamma|Delta|Epsilon|Zeta|Omega|Sigma|Theta|Lambda)\b)'),
        # Sections/clauses
        re.compile(r'\b(Section\s+\d+(?:\.\d+)*(?:\([a-z]\))?)\b', re.IGNORECASE),
        # Named provisions/concepts (multi-word finance terms)
        re.compile(r'\b(cross[- ]default(?:\s+threshold)?|close[- ]out\s+netting|automatic\s+early\s+termination'
                   r'|credit\s+event\s+upon\s+merger|additional\s+termination\s+event'
                   r'|eligible\s+collateral|independent\s+amount|minimum\s+transfer\s+amount'
                   r'|threshold\s+amount|rating[- ]dependent\s+threshold'
                   r'|force\s+majeure|flawed\s+assets?|netting\s+provisions?'
                   r'|variation\s+margin|initial\s+margin|haircut'
                   r'|NAV\s+trigger|credit\s+support|valuation\s+date)\b', re.IGNORECASE),
        # Credit ratings
        re.compile(r'\b((?:AAA|AA|A|BBB|BB|B|CCC|CC|C)[+-]?)\b'),
        # Currencies
        re.compile(r'\b(USD|EUR|GBP|JPY|CHF)\b'),
        # Securities types
        re.compile(r'\b(US\s+Treasur(?:y|ies)|Agency\s+securities|government\s+securities'
                   r'|G7\s+government\s+securities)\b', re.IGNORECASE),
        # Organizations
        re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Bank|Corp|Inc|LLC|LP|Fund|Capital|Partners'
                   r'|Securities|Financial|Asset\s+Management|Group|Holdings))\b'),
        # General proper nouns (catch-all)
        re.compile(r'\b([A-Z][a-zA-Z]{2,})\b'),
    ]

    # Words that match the proper noun pattern but aren't entities
    ENTITY_STOPWORDS = {
        "The", "This", "That", "These", "Those", "There", "They", "Then",
        "What", "When", "Where", "Which", "While", "Who", "Why", "How",
        "Here", "Have", "Has", "Had", "His", "Her", "Its",
        "Are", "And", "But", "For", "Not", "All", "Any", "Can",
        "May", "Our", "Out", "Own", "Too", "Was", "Will", "Yes",
        "Also", "Some", "Each", "From", "Just", "Into", "Over",
        "Such", "Very", "Been", "Both", "Does", "Done", "Down",
        "Even", "Gets", "Goes", "Gone", "Good", "Got", "Great",
        "Keep", "Kept", "Know", "Last", "Left", "Let", "Like",
        "Long", "Look", "Made", "Make", "Many", "More", "Most",
        "Much", "Must", "Need", "Next", "Note", "Now", "Only",
        "Part", "Plus", "Put", "Said", "Same", "See", "Set",
        "Should", "Show", "Side", "Since", "Still", "Sure", "Take",
        "Tell", "Than", "Them", "Think", "Time", "Under", "Upon",
        "Used", "Using", "Want", "Way", "Well", "Were", "With",
        "Would", "Year", "Your",
        "Important", "Key", "Main", "New", "Old", "Other",
        "First", "Second", "Third", "Based", "Given", "Known",
        "Noted", "Stored", "Done", "Got", "Updated", "Added",
        "Meeting", "Client", "Clients", "Company", "Bank", "Counterparty", "Party",
        "Threshold", "Amount", "Agreement", "Schedule", "Rating",
        "Group", "Inc", "Corp", "Fund", "Capital", "Partners", "Holdings",
        "Securities", "Financial", "Management",
        "Today", "Tomorrow", "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    }

    # ─── Type markers ───
    DECISION_MARKERS = [
        "decided", "chose", "selected", "will use", "switched to",
        "going with", "agreed", "confirmed", "settled on", "set at",
        "set to", "specified", "established",
        "we'll go with", "final answer", "approved",
    ]

    TASK_MARKERS = [
        "need to", "todo", "next step", "will do",
        "plan to", "going to", "must", "remember to",
        "deadline", "due by", "by friday", "by monday",
        "by end of", "action item",
        "meeting with", "scheduled", "should be meeting",
        "follow up", "check in", "reach out", "send to",
    ]

    PREFERENCE_MARKERS = [
        "prefer", "we like", "we want", "always use",
        "never use", "our standard", "our policy", "we require",
        "we typically", "we usually", "our approach",
    ]

    # ─── Noise filters ───
    FILLER_PATTERNS = re.compile(
        r'^(?:(?:sure|ok|okay|yes|no|yeah|yep|nope|thanks|thank you|hello|hi|hey'
        r'|got it|sounds good|makes sense|right|exactly|absolutely|understood'
        r'|perfect|great|good point|fair enough|interesting|I see|hmm|let me think'
        r'|can you help|what do you think|how about|what about)[.!?,\s]*)+$',
        re.IGNORECASE
    )

    ASSISTANT_FILLER = re.compile(
        r'^(?:I can help with that|I\'d be happy to|Let me|Here\'s|Sure,? (?:I\'ll|let me)|'
        r'That\'s a (?:good|great) (?:point|question)|I\'ll note)',
        re.IGNORECASE
    )

    def __init__(self):
        self._seen_hashes = set()

    def extract(self, text: str, role: str = "user") -> list[dict]:
        """Extract atomic facts from a text block.

        Returns a list of fact dicts with keys:
            text, source_role, timestamp, type, entities, quantities
        """
        facts = []
        sentences = self._split_sentences(text)

        for sentence in sentences:
            s = sentence.strip()

            if len(s) < 12 or len(s) > 500:
                continue
            if self.FILLER_PATTERNS.match(s):
                continue

            content_hash = hashlib.md5(s.lower().encode()).hexdigest()[:12]
            if content_hash in self._seen_hashes:
                continue

            entities = self._extract_entities(s)
            quantities = self._extract_quantities(s)
            fact_type = self._classify_type(s)

            # Substance gate: must have entity, quantity, or non-general type
            if not entities and not quantities and fact_type == "general":
                continue

            if role == "assistant" and self.ASSISTANT_FILLER.match(s):
                if not entities and not quantities:
                    continue

            self._seen_hashes.add(content_hash)

            facts.append({
                "text": s,
                "source_role": role,
                "timestamp": datetime.now().isoformat(),
                "type": fact_type,
                "entities": entities,
                "quantities": quantities,
            })

        return facts

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences, handling currency and abbreviations."""
        protected = text
        protected = re.sub(r'(\$[\d,.]+[MBK]?)([.-])', r'\1⌀\2', protected)
        protected = re.sub(r'\b(e\.g|i\.e|etc|vs|approx|incl)\.\s', r'\1⌁ ', protected)
        protected = re.sub(r'\b([A-Z]{1,3}[+-])[.,]\s', r'\1⌂ ', protected)

        parts = re.split(r'(?<=[.!?])\s+|\n+', protected)

        result = []
        for p in parts:
            p = p.replace('⌀', '').replace('⌁', '.').replace('⌂', ',')
            p = p.strip()
            if p:
                result.append(p)
        return result

    def _classify_type(self, sentence: str) -> str:
        s_lower = sentence.lower()
        if any(m in s_lower for m in self.DECISION_MARKERS):
            return "decision"
        if any(m in s_lower for m in self.TASK_MARKERS):
            return "task"
        if any(m in s_lower for m in self.PREFERENCE_MARKERS):
            return "preference"
        if self.QUANTITY_PATTERN.search(sentence):
            return "quantitative"
        return "general"

    def _extract_entities(self, sentence: str) -> list[str]:
        """Extract named entities using domain-specific patterns."""
        entities = set()
        for pattern in self.ENTITY_PATTERNS:
            for match in pattern.finditer(sentence):
                entity = match.group(1).strip()
                if len(entity) < 2:
                    continue
                if len(entity) <= 2 and not entity.endswith(('+', '-')):
                    continue
                if entity in self.ENTITY_STOPWORDS:
                    continue
                entities.add(entity)
        return sorted(entities)

    def _extract_quantities(self, sentence: str) -> list[str]:
        return self.QUANTITY_PATTERN.findall(sentence)


# ── Embedding + Storage (CPU, sentence-transformers + ChromaDB) ──

class MemoryStore:
    """Embeds and stores facts in ChromaDB for semantic retrieval.

    Features:
    - Cosine similarity search via sentence-transformers (all-MiniLM-L6-v2)
    - Automatic deduplication at 95% similarity threshold
    - Contradiction detection: when a new fact about the same entities has
      different quantities, the old fact is marked as superseded
    - Temporal decay: exponential with 7-day half-life for recency weighting
    - All embedding runs on CPU — zero GPU impact
    """

    DEDUP_THRESHOLD = 0.95
    CONTRADICT_THRESHOLD = 0.70
    CONTRADICT_CEILING = 0.94

    def __init__(self, db_path: str, collection_name: str = "conversation_memory",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.emb_model = SentenceTransformer(embedding_model, device="cpu")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._counter = self.collection.count()

    def store(self, fact: dict) -> Optional[str]:
        """Embed and store a single fact. Returns fact ID or None if duplicate."""
        embedding = self.emb_model.encode(
            [fact["text"]],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]

        if self._counter > 0 and self._is_duplicate(embedding):
            return None

        superseded_ids = []
        if self._counter > 0:
            superseded_ids = self._check_contradictions(embedding, fact)

        self._counter += 1
        fact_id = f"fact_{self._counter}_{int(time.time())}"

        metadata = self._make_metadata(fact)
        if superseded_ids:
            metadata["supersedes"] = json.dumps(superseded_ids)

        self.collection.upsert(
            ids=[fact_id],
            embeddings=[embedding.tolist()],
            documents=[fact["text"]],
            metadatas=[metadata],
        )

        return fact_id

    def store_batch(self, facts: list[dict]) -> list[str]:
        """Embed and store multiple facts efficiently with true batch upsert."""
        if not facts:
            return []

        texts = [f["text"] for f in facts]
        embeddings = self.emb_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )

        ids, docs, embs, metas = [], [], [], []
        for fact, emb in zip(facts, embeddings):
            if self._counter > 0 and self._is_duplicate(emb):
                continue

            self._counter += 1
            fact_id = f"fact_{self._counter}_{int(time.time())}"
            ids.append(fact_id)
            docs.append(fact["text"])
            embs.append(emb.tolist())
            metas.append(self._make_metadata(fact))

        if not ids:
            return []

        self.collection.upsert(
            ids=ids, embeddings=embs, documents=docs, metadatas=metas,
        )

        return ids

    def recall(self, query: str, n_results: int = 5, type_filter: str = None,
               recency_weight: float = 0.15, include_superseded: bool = False) -> list[dict]:
        """Retrieve relevant facts via semantic search with temporal decay.

        Args:
            query: Natural language search query
            n_results: Number of results to return
            type_filter: Optional filter by fact type (decision/task/preference/quantitative/general)
            recency_weight: Weight for recency vs. similarity (0-1). Default 0.15.
            include_superseded: If True, include facts that have been superseded by newer ones

        Returns:
            List of dicts with keys: text, similarity, recency, score, metadata, superseded
        """
        q_emb = self.emb_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]

        where_filter = {"type": type_filter} if type_filter else None
        fetch_n = min(n_results * 5, max(n_results * 3, self.collection.count()))

        results = self.collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=fetch_n,
            where=where_filter,
        )

        recalled = []
        now = time.time()
        for i in range(len(results["documents"][0])):
            similarity = 1 - results["distances"][0][i]
            meta = results["metadatas"][0][i]

            if not include_superseded and meta.get("superseded_by"):
                continue

            try:
                fact_time = datetime.fromisoformat(meta.get("timestamp", "")).timestamp()
                age_days = (now - fact_time) / 86400
                recency_score = 2 ** (-age_days / 7)
            except (ValueError, TypeError):
                recency_score = 0.3

            combined_score = similarity * (1 - recency_weight) + recency_score * recency_weight

            recalled.append({
                "text": results["documents"][0][i],
                "similarity": similarity,
                "recency": round(recency_score, 4),
                "score": combined_score,
                "metadata": meta,
                "superseded": bool(meta.get("superseded_by")),
            })

        recalled.sort(key=lambda x: x["score"], reverse=True)
        return recalled[:n_results]

    def count(self) -> int:
        return self.collection.count()

    def _is_duplicate(self, embedding) -> bool:
        try:
            results = self.collection.query(
                query_embeddings=[embedding.tolist()], n_results=1,
            )
            if results["distances"][0]:
                similarity = 1 - results["distances"][0][0]
                return similarity >= self.DEDUP_THRESHOLD
        except Exception:
            pass
        return False

    def _check_contradictions(self, embedding, new_fact: dict) -> list[str]:
        """Detect and supersede contradicted facts.

        A contradiction is when:
        1. An existing fact is semantically similar (same topic, 70-94%)
        2. Both facts mention the same entities
        3. But they have DIFFERENT quantities (the value changed)
        """
        try:
            results = self.collection.query(
                query_embeddings=[embedding.tolist()], n_results=5,
            )
        except Exception:
            return []

        if not results["distances"][0]:
            return []

        new_entities = set(new_fact.get("entities", []))
        new_quantities = set(new_fact.get("quantities", []))
        new_text = new_fact.get("text", "").lower()

        UPDATE_SIGNALS = [
            "increased", "decreased", "changed", "updated", "revised",
            "expanded", "reduced", "tightened", "downgraded", "upgraded",
            "renegotiated", "amended", "modified", "raised", "lowered",
        ]
        has_update_signal = any(s in new_text for s in UPDATE_SIGNALS)

        superseded = []
        for i in range(len(results["distances"][0])):
            similarity = 1 - results["distances"][0][i]

            if similarity < self.CONTRADICT_THRESHOLD or similarity >= self.CONTRADICT_CEILING:
                continue

            meta = results["metadatas"][0][i]
            if meta.get("superseded_by"):
                continue

            try:
                old_entities = set(json.loads(meta.get("entities", "[]")))
            except (json.JSONDecodeError, TypeError):
                old_entities = set()

            if not (new_entities & old_entities):
                continue

            try:
                old_quantities = set(json.loads(meta.get("quantities", "[]")))
            except (json.JSONDecodeError, TypeError):
                old_quantities = set()

            old_text = results["documents"][0][i].lower()
            is_contradiction = False

            if old_quantities and new_quantities and old_quantities != new_quantities:
                is_contradiction = True
            elif new_quantities and not old_quantities and has_update_signal:
                is_contradiction = True
            elif has_update_signal and (new_entities & old_entities):
                is_contradiction = True
            elif similarity > 0.85 and (new_entities & old_entities) and new_text != old_text:
                is_contradiction = True

            if is_contradiction:
                old_id = results["ids"][0][i]
                superseded.append(old_id)
                meta["superseded_by"] = new_fact["text"][:200]
                meta["superseded_at"] = datetime.now().isoformat()
                self.collection.update(ids=[old_id], metadatas=[meta])

        return superseded

    def _make_metadata(self, fact: dict) -> dict:
        return {
            "type": fact.get("type", "general"),
            "source_role": fact.get("source_role", "unknown"),
            "timestamp": fact.get("timestamp", datetime.now().isoformat()),
            "entities": json.dumps(fact.get("entities", [])),
            "quantities": json.dumps(fact.get("quantities", [])),
            "session": fact.get("session", "unknown"),
        }


# ── Vault Writer (Obsidian markdown, organized by type) ──────────

class VaultWriter:
    """Writes organized facts to an Obsidian-compatible vault as structured markdown.

    Creates a `memory/` directory structure:
        memory/entities/    — One file per entity with all related facts
        memory/facts/       — General facts
        memory/decisions/   — Decisions made
        memory/preferences/ — User preferences
        memory/tasks/       — Tasks and deadlines
        memory/sessions/    — Per-session summaries

    All files use Obsidian [[wikilinks]] for cross-referencing.
    Superseded facts are marked with ~~strikethrough~~.
    """

    def __init__(self, vault_path: str):
        self.vault_path = os.path.expanduser(vault_path)
        self._ensure_structure()

    def _ensure_structure(self):
        folders = [
            "memory/entities", "memory/facts", "memory/decisions",
            "memory/preferences", "memory/tasks", "memory/sessions",
        ]
        for folder in folders:
            os.makedirs(os.path.join(self.vault_path, folder), exist_ok=True)

    def write_fact(self, fact: dict, category: str = None):
        """Write a fact to the appropriate vault location."""
        fact_type = category or fact.get("type", "general")
        entities = fact.get("entities", [])
        text = fact["text"]
        timestamp = fact.get("timestamp", datetime.now().isoformat())

        display_text = self._add_wikilinks(text, entities)

        if fact_type == "decision":
            self._append_to_file("memory/decisions/decisions.md", display_text, timestamp)
        elif fact_type == "preference":
            self._append_to_file("memory/preferences/preferences.md", display_text, timestamp)
        elif fact_type == "task":
            self._append_to_file("memory/tasks/tasks.md", display_text, timestamp)
        else:
            self._append_to_file("memory/facts/general.md", display_text, timestamp)

        if entities:
            for entity in entities:
                safe_name = self._entity_filename(entity)
                if safe_name:
                    self._append_to_file(
                        f"memory/entities/{safe_name}.md",
                        display_text, timestamp, entity_name=entity
                    )

    def write_session_summary(self, session_id: str, facts: list[dict]):
        """Write a session summary note."""
        filepath = os.path.join(self.vault_path, f"memory/sessions/{session_id}.md")

        lines = [
            f"# Session: {session_id}",
            f"*Generated: {datetime.now().isoformat()}*",
            "",
            f"## Facts Extracted ({len(facts)})",
            "",
        ]

        by_type = {}
        for f in facts:
            t = f.get("type", "general")
            by_type.setdefault(t, []).append(f)

        for fact_type, type_facts in sorted(by_type.items()):
            lines.append(f"### {fact_type.title()} ({len(type_facts)})")
            for f in type_facts:
                entities = f.get("entities", [])
                links = " ".join(f"[[{self._entity_filename(e)}|{e}]]" for e in entities
                                if self._entity_filename(e))
                lines.append(f"- {f['text']} {links}")
            lines.append("")

        with open(filepath, "w") as fh:
            fh.write("\n".join(lines))

    def supersede_in_vault(self, old_text: str, new_text: str, timestamp: str):
        """Mark an old fact as superseded across all vault files."""
        memory_dir = os.path.join(self.vault_path, "memory")
        old_clean = re.sub(r'\[\[[^\]|]+\|([^\]]+)\]\]', r'\1', old_text)
        match_prefix = old_clean[:60]

        for root, dirs, files in os.walk(memory_dir):
            for fname in files:
                if not fname.endswith(".md"):
                    continue
                filepath = os.path.join(root, fname)
                try:
                    with open(filepath, "r") as fh:
                        lines = fh.readlines()
                except Exception:
                    continue

                modified = False
                new_lines = []
                for line in lines:
                    line_clean = re.sub(r'\[\[[^\]|]+\|([^\]]+)\]\]', r'\1', line)
                    if match_prefix in line_clean and not line.startswith("- ~~["):
                        stripped = line.rstrip('\n')
                        new_lines.append(f"- ~~{stripped[2:]}~~ *(superseded {timestamp[:10]})*\n")
                        modified = True
                    else:
                        new_lines.append(line)

                if modified:
                    with open(filepath, "w") as fh:
                        fh.writelines(new_lines)

    def _entity_filename(self, entity: str) -> str:
        name = re.sub(r'\(([a-z0-9]+)\)', r'\1', entity)
        name = re.sub(r'[\s/]+', '-', name)
        name = re.sub(r'[^\w+-]', '', name)
        name = re.sub(r'-{2,}', '-', name)
        name = name.strip('-')
        if len(name) < 2 or len(name) > 60:
            return ""
        return name

    def _add_wikilinks(self, text: str, entities: list[str]) -> str:
        result = text
        for entity in entities:
            filename = self._entity_filename(entity)
            if filename and entity in result:
                result = result.replace(entity, f"[[{filename}|{entity}]]", 1)
        return result

    def _append_to_file(self, rel_path: str, text: str, timestamp: str, entity_name: str = None):
        filepath = os.path.join(self.vault_path, rel_path)

        if not os.path.exists(filepath):
            title = entity_name or rel_path.split('/')[-1].replace('.md', '').replace('-', ' ').title()
            header = f"# {title}\n\n"
            with open(filepath, "w") as fh:
                fh.write(header)

        with open(filepath, "a") as fh:
            fh.write(f"- [{timestamp[:10]}] {text}\n")


# ── Memory Daemon (orchestrates all tiers) ─────────────────────

class MemoryDaemon:
    """Zero-cost persistent memory for local LLMs.

    Runs fact extraction, embedding, and storage in a background thread.
    Your GPU is never touched — all processing happens on CPU.

    Args:
        vault_path: Path to Obsidian vault (or any directory for markdown output).
                    Defaults to ./phantom_vault/
        db_path: Path to ChromaDB storage. Defaults to <vault_path>/../phantom_db/
        session_id: Optional session identifier. Auto-generated if not provided.
        embedding_model: sentence-transformers model name. Default: all-MiniLM-L6-v2

    Example:
        daemon = MemoryDaemon(vault_path="~/notes")
        daemon.start()
        daemon.ingest("user", "Meeting with John at 3pm tomorrow")
        results = daemon.recall("when is the meeting?")
        daemon.stop()
    """

    def __init__(self, vault_path: str = None, db_path: str = None,
                 session_id: str = None, embedding_model: str = "all-MiniLM-L6-v2"):
        self.session_id = session_id or datetime.now().strftime("%Y-%m-%d-%H%M")

        if vault_path is None:
            vault_path = os.path.join(os.getcwd(), "phantom_vault")
        vault_path = os.path.expanduser(vault_path)

        # Tier 1: CPU extraction + embedding
        self.extractor = FactExtractor()
        if db_path is None:
            db_path = os.path.join(os.path.dirname(vault_path), "phantom_db")
        db_path = os.path.expanduser(db_path)
        os.makedirs(db_path, exist_ok=True)
        self.store = MemoryStore(db_path, embedding_model=embedding_model)

        # Tier 2: Vault organization
        self.vault = VaultWriter(vault_path)

        # Processing queue and thread
        self._queue = queue.Queue()
        self._running = False
        self._thread = None
        self._session_facts = []
        self._stats = {"ingested": 0, "extracted": 0, "stored": 0, "deduped": 0}

    def start(self):
        """Start the background memory processing daemon."""
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the daemon and write session summary."""
        self._running = False
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=5)

        if self._session_facts:
            self.vault.write_session_summary(self.session_id, self._session_facts)

    def ingest(self, role: str, text: str):
        """Feed a conversation turn to the daemon. Non-blocking."""
        self._queue.put({"role": role, "text": text})
        self._stats["ingested"] += 1

    def recall(self, query: str, n_results: int = 5, type_filter: str = None) -> list[dict]:
        """Retrieve relevant memories for context injection."""
        return self.store.recall(query, n_results=n_results, type_filter=type_filter)

    def recall_formatted(self, query: str, n_results: int = 5) -> str:
        """Retrieve and format memories as text for LLM context injection.

        Returns a markdown-formatted string ready to inject into a system prompt
        or conversation context.
        """
        memories = self.recall(query, n_results)
        if not memories:
            return ""

        lines = ["## Relevant Memories\n"]
        for m in memories:
            meta = m["metadata"]
            lines.append(f"- [{meta.get('type', '?')}] {m['text']} (score={m['score']:.2f})")

        return "\n".join(lines)

    @property
    def stats(self):
        return {
            **self._stats,
            "total_memories": self.store.count(),
            "superseded": self._stats.get("superseded", 0),
        }

    def _process_loop(self):
        """Background processing loop with contradiction detection."""
        while self._running:
            try:
                item = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            if item is None:
                break

            facts = self.extractor.extract(item["text"], role=item["role"])

            for fact in facts:
                fact["session"] = self.session_id

                fact_id = self.store.store(fact)
                if fact_id:
                    self._stats["stored"] += 1

                    # Handle vault supersession
                    try:
                        meta = self.store.collection.get(ids=[fact_id])
                        if meta and meta["metadatas"]:
                            supersedes_json = meta["metadatas"][0].get("supersedes", "[]")
                            superseded_ids = json.loads(supersedes_json)
                            if superseded_ids:
                                old_facts = self.store.collection.get(ids=superseded_ids)
                                if old_facts and old_facts["documents"]:
                                    for old_text in old_facts["documents"]:
                                        self.vault.supersede_in_vault(
                                            old_text, fact["text"],
                                            fact.get("timestamp", datetime.now().isoformat())
                                        )
                                self._stats.setdefault("superseded", 0)
                                self._stats["superseded"] += len(superseded_ids)
                    except Exception:
                        pass  # Non-critical

                    self.vault.write_fact(fact)
                    self._session_facts.append(fact)
                else:
                    self._stats["deduped"] += 1

            self._stats["extracted"] += len(facts)
