#!/usr/bin/env python3
"""Greedy outputs must not depend on the drafter: the target model verifies every token.
  collect NAME PORT      -> /root/tune/night4/greedy-NAME.json (16 prompts, temperature 0, thinking off, 256 tokens)
  compare A B            -> how many of the 16 answers are identical (expect 16, near-ties may flip 1-2)"""
import json, os, sys, urllib.request
from pathlib import Path

D = Path("/root/tune/night4")
PROMPTS = [
    "Write a Python function that merges two sorted lists.",
    "Explain what a mutex is in three sentences.",
    "Write a bash one-liner that counts lines in all .py files recursively.",
    "Refactor: def f(x):\n    if x == 1: return 'a'\n    elif x == 2: return 'b'\n    else: return 'c'",
    "What is the capital of Norway and why is it located where it is?",
    "Write a SQL query for the top 5 customers by total order value.",
    "Translate to Danish: The server restarts every night at two o'clock.",
    "Write a TypeScript interface for a user with id, name, email and optional roles.",
    "List the planets of the solar system in order.",
    "Write a Rust function that returns the nth Fibonacci number iteratively.",
    "Summarize the CAP theorem.",
    "Write a regular expression that matches an ISO 8601 date.",
    "Write a Dockerfile for a small Python Flask app.",
    "What does 'git rebase -i' do?",
    "Write a JSON object describing a book with title, author and year.",
    "Explain big-O of binary search.",
]


def key():
    return os.environ.get("API_KEY") or Path(os.environ.get("KEY_FILE", "/root/llm-api-key")).read_text().strip()


def ask(port, p):
    body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": p}], "max_tokens": 256,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"].get("content") or ""


if sys.argv[1] == "collect":
    name, port = sys.argv[2], int(sys.argv[3])
    (D / f"greedy-{name}.json").write_text(json.dumps([ask(port, p) for p in PROMPTS]))
    print(f"greedy {name}: collected {len(PROMPTS)}")
else:
    a = json.loads((D / f"greedy-{sys.argv[2]}.json").read_text()); b = json.loads((D / f"greedy-{sys.argv[3]}.json").read_text())
    same = sum(x == y for x, y in zip(a, b))
    first = [next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y))) for x, y in zip(a, b) if x != y]
    print(f"greedy identity {sys.argv[2]} vs {sys.argv[3]}: {same}/{len(a)} identical; first difference at chars {first}")
