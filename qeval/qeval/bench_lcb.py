"""LiveCodeBench code generation (release v6 window, optionally v5), hidden tests executed."""
import gzip
import json
import os

from . import sandbox
from .common import DATA, extract_code, read_jsonl

SYSTEM = ("You are an expert Python programmer. You will be given a question (problem specification) "
          "and will generate a correct Python program that matches the specification and passes all tests.")
FMT_STARTER = ("You will use the following starter code to write the solution to the problem and "
               "enclose your code within delimiters.")
FMT_STDIN = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly "
             "test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when "
             "the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.")


def prompt_for(p):
    s = f"### Question:\n{p['question_content']}\n\n"
    if p.get("starter_code"):
        s += f"### Format: {FMT_STARTER}\n```python\n{p['starter_code']}\n```\n\n"
    else:
        s += f"### Format: {FMT_STDIN}\n```python\n# YOUR CODE HERE\n```\n\n"
    s += "### Answer: (use the provided format with backticks)\n\n"
    return s


def load(releases=("v6",), difficulties=None):
    items = []
    for rel in releases:
        for p in read_jsonl(os.path.join(DATA, "lcb", f"problems_{rel}.jsonl")):
            if difficulties and p["difficulty"] not in difficulties:
                continue
            items.append({"id": p["question_id"], "subset": f"lcb_{rel}_{p['difficulty']}",
                          "difficulty": p["difficulty"], "platform": p["platform"], "problem": p})
    return items


def make_fn(cfg):
    def fn(item, ctx):
        p = item["problem"]
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt_for(p)}]
        r = ctx.chat(msgs, max_tokens=cfg.max_tokens)
        base = {"difficulty": item["difficulty"], "platform": item["platform"]}
        if r.get("error"):
            return dict(base, correct=False, status="api_error")
        code = extract_code(r["content"])
        if not code.strip():
            return dict(base, correct=False, status="no_code", answer=r["content"][-2000:])
        with gzip.open(os.path.join(DATA, "lcb", "tests_" + p.get("release", "v6"), p["question_id"] + ".json.gz"), "rt") as f:
            tests = json.load(f)
        n = len(tests["inputs"])
        timeout = min(cfg.exec_timeout or 900, (6 + 1) * n + 5 + 10)
        res = sandbox.run_worker("worker_lcb.py", {"code": code, "tests": tests, "timeout": 6}, timeout=timeout)
        meta = res.get("meta") or {}
        return dict(base, correct=bool(res.get("passed")), status="pass" if res.get("passed") else (
            meta.get("error_message") or res.get("status") or "fail"), n_tests=n, n_run=res.get("n_run"),
            exec_s=res.get("exec_s"), sandbox_timeout=res.get("sandbox_timeout"), code=code[-8000:])
    return fn
