"""HumanEval+ (164) and MBPP+ (378) from EvalPlus, code executed in the sandbox."""
import os

from . import sandbox
from .common import DATA, extract_code, read_jsonl

INSTRUCTION = ("Please provide a self-contained Python script that solves the following problem "
               "in a markdown code block:")  # evalplus OpenAI-backend prompt
FILES = {"humaneval": "HumanEvalPlus-v0.1.10.jsonl.gz", "mbpp": "MbppPlus-v0.2.0.jsonl.gz"}


def load(subsets=("humaneval", "mbpp")):
    items = []
    for ds in subsets:
        for t in read_jsonl(os.path.join(DATA, "evalplus", FILES[ds])):
            items.append({"id": t["task_id"], "subset": ds + "+", "dataset": ds, "task": t})
    return items


def _imports_from_prompt(prompt):
    out = []
    for line in prompt.splitlines():
        s = line.strip()
        if (s.startswith("import ") or s.startswith("from ")) and line == s:
            out.append(s)
    return "\n".join(out)


def make_fn(cfg):
    def fn(item, ctx):
        t = item["task"]
        msg = INSTRUCTION + f"\n```python\n{t['prompt'].strip()}\n```"
        r = ctx.chat([{"role": "user", "content": msg}], max_tokens=cfg.max_tokens)
        if r.get("error"):
            return {"correct": False, "status": "api_error"}
        code = extract_code(r["content"], must_contain=f"def {t['entry_point']}")
        if not code.strip():
            return {"correct": False, "status": "no_code", "answer": r["content"][-2000:]}
        solution = code
        if item["dataset"] == "humaneval":
            imps = _imports_from_prompt(t["prompt"])
            if imps:
                solution = imps + "\n" + code
        payload = {
            "dataset": item["dataset"], "task_id": t["task_id"], "entry_point": t["entry_point"],
            "canonical": t["prompt"] + t["canonical_solution"], "solution": solution,
            "base_input": t["base_input"], "plus_input": t["plus_input"], "atol": t.get("atol", 0),
        }
        res = sandbox.run_worker("worker_evalplus.py", payload, timeout=cfg.exec_timeout or 300)
        base_ok = res.get("base") == "pass"
        plus_ok = base_ok and res.get("plus") == "pass"
        return {"correct": plus_ok, "base_correct": base_ok, "status": "pass" if plus_ok else (
            res.get("fail_phase") or res.get("status") or "fail"), "exec": {k: res.get(k) for k in (
                "base", "plus", "fail_phase", "fail_index", "error", "sandbox_timeout")},
                "code": solution[-6000:]}
    return fn
