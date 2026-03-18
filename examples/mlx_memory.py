#!/usr/bin/env python3
"""
Add persistent memory to mlx-lm server.

Prerequisites:
    pip install phantom-memory openai
    mlx_lm.server --model mlx-community/Qwen3.5-9B-MLX-4bit --port 8899

Usage:
    python mlx_memory.py
"""

from openai import OpenAI
from phantom import MemoryDaemon

# Start the memory daemon
daemon = MemoryDaemon(vault_path="./my_vault")
daemon.start()

# Connect to mlx-lm server (OpenAI-compatible)
client = OpenAI(base_url="http://127.0.0.1:8899/v1", api_key="not-needed")
MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"
history = []

print("Chat with MLX + persistent memory. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit"):
        break

    # Inject relevant memories
    memory_context = daemon.recall_formatted(user_input)

    messages = []
    if memory_context:
        messages.append({"role": "system", "content": memory_context})
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    # Call MLX server
    response = client.chat.completions.create(model=MODEL, messages=messages)
    reply = response.choices[0].message.content
    print(f"\nAssistant: {reply}\n")

    # Store in memory (non-blocking)
    daemon.ingest("user", user_input)
    daemon.ingest("assistant", reply)

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})

daemon.stop()
print(f"\nSession saved. {daemon.stats['stored']} facts stored.")
