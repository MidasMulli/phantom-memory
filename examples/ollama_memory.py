#!/usr/bin/env python3
"""
Add persistent memory to Ollama in 10 lines.

Prerequisites:
    pip install phantom-memory
    ollama pull llama3.1:8b  # or any model

Usage:
    python ollama_memory.py
"""

import requests
from phantom import MemoryDaemon

# Start the memory daemon (background thread, zero GPU impact)
daemon = MemoryDaemon(vault_path="./my_vault")
daemon.start()

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"
history = []

print("Chat with Ollama + persistent memory. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit"):
        break

    # Inject relevant memories into context
    memory_context = daemon.recall_formatted(user_input)

    # Build messages with memory context
    messages = []
    if memory_context:
        messages.append({"role": "system", "content": memory_context})
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    # Call Ollama
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": messages,
        "stream": False,
    })
    reply = response.json()["message"]["content"]
    print(f"\nAssistant: {reply}\n")

    # Store the conversation turn in memory (non-blocking, background)
    daemon.ingest("user", user_input)
    daemon.ingest("assistant", reply)

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})

# Stop daemon and write session summary
daemon.stop()
print(f"\nSession saved. {daemon.stats['stored']} facts stored to memory.")
