#!/usr/bin/env python3
"""
Add persistent memory to ANY OpenAI-compatible server.

Works with: Ollama, mlx-lm, LM Studio, vLLM, text-generation-inference,
LocalAI, llama.cpp server, or any server that speaks the OpenAI chat API.

Prerequisites:
    pip install phantom-memory openai

Usage:
    # Point to your local server
    export LLM_BASE_URL=http://localhost:11434/v1   # Ollama
    export LLM_MODEL=llama3.1:8b
    python openai_compat.py
"""

import os
from openai import OpenAI
from phantom import MemoryDaemon

# ── Configuration ──
BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("LLM_MODEL", "llama3.1:8b")
API_KEY = os.environ.get("LLM_API_KEY", "not-needed")
VAULT_PATH = os.environ.get("PHANTOM_VAULT", "./phantom_vault")

# ── Start memory daemon ──
daemon = MemoryDaemon(vault_path=VAULT_PATH)
daemon.start()
print(f"◈ Phantom Memory active ({daemon.store.count()} memories)")
print(f"  Model: {MODEL} @ {BASE_URL}")
print(f"  Vault: {VAULT_PATH}\n")

# ── Connect to LLM ──
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
history = []

while True:
    user_input = input("You: ").strip()
    if not user_input or user_input.lower() in ("quit", "exit", "q"):
        break

    # Recall relevant memories and inject as system context
    memory_context = daemon.recall_formatted(user_input, n_results=5)

    messages = []
    if memory_context:
        messages.append({"role": "system", "content": memory_context})
    messages.extend(history[-20:])  # Keep last 20 messages for context window
    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(model=MODEL, messages=messages)
        reply = response.choices[0].message.content
    except Exception as e:
        print(f"\n  Error: {e}\n")
        continue

    print(f"\nAssistant: {reply}\n")

    # Store both sides (non-blocking, background thread)
    daemon.ingest("user", user_input)
    daemon.ingest("assistant", reply)

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})

# ── Shutdown ──
daemon.stop()
stats = daemon.stats
print(f"\n◈ Session complete: {stats['stored']} facts stored, {stats['deduped']} deduped")
