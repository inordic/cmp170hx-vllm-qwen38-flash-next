#!/usr/bin/env python3
"""qeval: overnight quality evals against an OpenAI-compatible endpoint (stdlib only, Python >= 3.10).

Examples (see B_quality_eval.md and run_tonight.sh):
  python3 run_eval.py selftest
  python3 run_eval.py probe   --endpoint http://127.0.0.1:8080 --key-file /root/llm-api-key --model qwen3.8-flash-next
  python3 run_eval.py run --bench evalplus --tag swift15 --effort medium --concurrency 16 ...
  python3 run_eval.py compare results/swift15/medium results/intel/medium results/cyankiwi/medium
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qeval import common, sandbox  # noqa: E402
from qeval.common import log  # noqa: E402

DEFAULT_MAX_TOKENS = {"evalplus": 16384, "lcb": 24576, "bfcl": 8192, "longctx": 8192}
DEFAULT_BFCL_SPEC = ("simple_python:100,multiple:100,parallel:100,parallel_multiple:100,irrelevance:100,"
                     "live_simple:100,live_multiple:150,live_parallel:16,live_parallel_multiple:24,"
                     "live_irrelevance:150,multi_turn_base:100")


class Cfg:
    pass


def add_common(ap):
    g = ap.add_argument_group("endpoint")
    g.add_argument("--endpoint", default=os.environ.get("ENDPOINT", "http://127.0.0.1:8080"))
    g.add_argument("--key", default=os.environ.get("KEY") or os.environ.get("QEVAL_KEY"))
    g.add_argument("--key-file", default=os.environ.get("KEY_FILE"))
    g.add_argument("--model", default=os.environ.get("MODEL", "qwen3.8-flash-next"))
    g.add_argument("--tag", default=os.environ.get("TAG"), help="label for results (e.g. swift15, intel, cyankiwi)")
    g.add_argument("--insecure", action="store_true")
    g = ap.add_argument_group("sampling")
    g.add_argument("--effort", default=os.environ.get("REASONING_EFFORT", "medium"),
                   help="reasoning_effort sent to the chat template; 'none' = send nothing (server default)")
    g.add_argument("--effort-mode", default="kwargs", choices=["kwargs", "top", "both"],
                   help="kwargs: chat_template_kwargs.reasoning_effort; top: top-level reasoning_effort")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--top-k", type=int, default=20)
    g.add_argument("--min-p", type=float, default=0.0)
    g.add_argument("--presence-penalty", type=float, default=0.0)
    g.add_argument("--seed-base", type=int, default=1234)
    g.add_argument("--no-seed", action="store_true", help="do not send per-request seeds")
    g.add_argument("--order-seed", type=int, default=7, help="item order; keep identical across models")
    g.add_argument("--max-tokens", type=int, default=None, help="per call; default depends on bench")
    g.add_argument("--no-stream", action="store_true")
    g.add_argument("--extra-body", default=None, help="JSON merged into every request body")
    g.add_argument("--history-reasoning", default="both", choices=["both", "reasoning_content", "reasoning", "none"],
                   help="how earlier reasoning is sent back in multi-step tool loops")
    g = ap.add_argument_group("run")
    g.add_argument("--concurrency", type=int, default=int(os.environ.get("CONCURRENCY", "8")))
    g.add_argument("--read-timeout", type=float, default=1800)
    g.add_argument("--retries", type=int, default=3)
    g.add_argument("--exec-workers", type=int, default=3, help="parallel sandbox executions")
    g.add_argument("--exec-timeout", type=float, default=None)
    g.add_argument("--sandbox-python", default=None)
    g.add_argument("--out", default=os.environ.get("OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")))


def make_cfg(a, bench=None):
    c = Cfg()
    c.endpoint, c.model, c.insecure = a.endpoint, a.model, a.insecure
    key = a.key
    if not key and a.key_file:
        with open(a.key_file) as f:
            key = f.read().strip()
    c.key = key
    c.tag = a.tag or a.model
    c.effort, c.effort_mode = a.effort, a.effort_mode
    c.temperature, c.top_p, c.top_k, c.min_p, c.presence_penalty = a.temperature, a.top_p, a.top_k, a.min_p, a.presence_penalty
    c.seed_base, c.use_seed, c.order_seed = a.seed_base, not a.no_seed, a.order_seed
    c.max_tokens = a.max_tokens or DEFAULT_MAX_TOKENS.get(bench, 8192)
    c.stream = not a.no_stream
    c.extra_body = (json.load(open(a.extra_body[1:])) if a.extra_body.startswith("@") else json.loads(a.extra_body)) \
        if a.extra_body else None
    c.history_reasoning = a.history_reasoning
    c.concurrency, c.read_timeout, c.retries = a.concurrency, a.read_timeout, a.retries
    c.exec_timeout = a.exec_timeout
    return c


def cfg_public(c):
    d = dict(vars(c))
    d.pop("key", None)
    return d


def cmd_selftest(a):
    sandbox.setup(a.sandbox_python, exec_workers=2)
    ok = True
    from qeval import bench_evalplus
    items = {it["id"]: it for it in bench_evalplus.load()}
    # (HumanEval/32 find_zero is excluded: its own canonical solution misses the 1e-4 oracle on one plus input)
    for tid in ("HumanEval/0", "HumanEval/10", "HumanEval/53", "HumanEval/129", "Mbpp/2", "Mbpp/124", "Mbpp/581", "Mbpp/722"):
        t = items[tid]["task"]
        ds = items[tid]["dataset"]
        sol = t["prompt"] + t["canonical_solution"]
        r = sandbox.run_worker("worker_evalplus.py", {"dataset": ds, "task_id": tid, "entry_point": t["entry_point"],
                                                      "canonical": sol, "solution": sol, "base_input": t["base_input"],
                                                      "plus_input": t["plus_input"], "atol": t.get("atol", 0)}, timeout=300)
        good = r.get("base") == "pass" and r.get("plus") == "pass"
        ok &= good
        log(f"selftest evalplus canonical {tid}: {'OK' if good else 'FAIL ' + json.dumps(r)}")
    t = items["HumanEval/53"]["task"]
    bad = "def add(x, y):\n    return x - y\n"
    r = sandbox.run_worker("worker_evalplus.py", {"dataset": "humaneval", "task_id": "HumanEval/53", "entry_point": "add",
                                                  "canonical": t["prompt"] + t["canonical_solution"], "solution": bad,
                                                  "base_input": t["base_input"], "plus_input": t["plus_input"], "atol": 0}, timeout=120)
    good = r.get("base") == "fail"
    ok &= good
    log(f"selftest evalplus wrong solution rejected: {'OK' if good else 'FAIL ' + json.dumps(r)}")
    cases = [
        ("stdin ok", "import sys\na,b=map(int,sys.stdin.read().split())\nprint(a+b)\n",
         {"inputs": ["1 2\n", "10 -3\n"], "outputs": ["3\n", "7\n"], "fn_name": None}, True),
        ("stdin input()", "n=int(input())\nprint(' '.join(str(i*i) for i in range(n)))\n",
         {"inputs": ["4\n"], "outputs": ["0 1 4 9\n"], "fn_name": None}, True),
        ("stdin wrong", "print(42)\n", {"inputs": ["1 2\n"], "outputs": ["3\n"], "fn_name": None}, False),
        ("stdin tle", "while True:\n    pass\n", {"inputs": ["1\n"], "outputs": ["1\n"], "fn_name": None}, False),
        ("call ok", "class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n"
                    "        d={}\n        for i,x in enumerate(nums):\n            if target-x in d: return [d[target-x],i]\n"
                    "            d[x]=i\n",
         {"inputs": ["[2,7,11,15]\n9", "[3,2,4]\n6"], "outputs": ["[0, 1]", "[1, 2]"], "fn_name": "twoSum"}, True),
    ]
    for name, code, tests, expect in cases:
        r = sandbox.run_worker("worker_lcb.py", {"code": code, "tests": tests, "timeout": 2}, timeout=60)
        good = bool(r.get("passed")) == expect
        ok &= good
        log(f"selftest lcb {name}: {'OK' if good else 'FAIL ' + json.dumps(r)}")
    # isolation checks: identity, network, write outside the work dir
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sandbox_escape_test")
    probe = ("import os,socket,json\nres={'uid':os.getuid()}\n"
             "try:\n s=socket.create_connection(('1.1.1.1',53),timeout=3); res['net']='OPEN'\nexcept Exception as e: res['net']='blocked'\n"
             f"try:\n fd=os.open({target!r}, os.O_WRONLY|os.O_CREAT, 0o600); os.close(fd); res['harness_dir_write']='ALLOWED'\n"
             "except Exception as e: res['harness_dir_write']='blocked'\n"
             "print(json.dumps(res))\n")
    r = sandbox.run_worker("worker_lcb.py", {"code": probe, "tests": {"inputs": [""], "outputs": ["x"], "fn_name": None},
                                             "timeout": 6}, timeout=30)
    log(f"selftest isolation probe (the output shows what sandboxed code could do): "
        f"{(r.get('meta') or {}).get('output')}")
    if os.path.exists(target):
        log(f"WARNING: sandboxed code could write into the harness dir ({target}); removing it")
        os.unlink(target)
    st = sandbox.setup()
    log(f"sandbox mode: {st['mode']} python={st['python']}")
    log("SELFTEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def cmd_probe(a):
    c = make_cfg(a, "probe")
    cl = common.ChatClient(c)
    out = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "cfg": cfg_public(c)}
    try:
        st, models = cl.get_json("/models")
        out["models"] = models
        log(f"/v1/models -> {st}: {[m.get('id') for m in models.get('data', [])]}")
    except Exception as e:
        log(f"/v1/models failed: {e}")
    q = [{"role": "user", "content": "How many prime numbers are there below 60? Answer with just the number."}]
    for eff in ("low", c.effort, "high"):
        c2 = make_cfg(a, "probe")
        c2.effort = eff
        r = common.ChatClient(c2).chat(q, max_tokens=8192, seed=1)
        u = r.get("usage") or {}
        out[f"effort_{eff}"] = {"completion_tokens": u.get("completion_tokens"), "finish": r.get("finish_reason"),
                                "reasoning_chars": len(r["reasoning"]), "content": r["content"][-200:],
                                "ttft_s": r.get("ttft_s"), "latency_s": r.get("latency_s"), "error": r.get("error")}
        log(f"effort={eff}: tokens={u.get('completion_tokens')} reasoning_chars={len(r['reasoning'])} "
            f"answer={r['content'].strip()[-40:]!r} err={r.get('error')}")
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
                                               "parameters": {"type": "object", "properties": {
                                                   "city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                                                   "required": ["city"]}}}]
    r = cl.chat([{"role": "user", "content": "What's the weather in Gothenburg in celsius?"}], tools=tools, max_tokens=4096, seed=1)
    out["tool_call"] = {"tool_calls": r["tool_calls"], "finish": r.get("finish_reason"), "error": r.get("error")}
    log(f"tool call: {r['tool_calls']} finish={r.get('finish_reason')} err={r.get('error')}")
    lo = (out.get("effort_low") or {}).get("completion_tokens") or 0
    hi = (out.get("effort_high") or {}).get("completion_tokens") or 0
    if lo and hi and lo >= hi:
        log("WARNING: low effort did not use fewer tokens than high - check that reasoning_effort reaches the template")
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, f"probe-{c.tag}-{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    log(f"probe written to {p}")
    return 0


def cmd_run(a):
    bench = a.bench
    c = make_cfg(a, bench)
    out_dir = os.path.join(a.out, c.tag, c.effort if c.effort else "default")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{bench}{a.suffix}.jsonl")
    if bench in ("evalplus", "lcb"):
        sandbox.setup(a.sandbox_python, exec_workers=a.exec_workers)
    if bench == "evalplus":
        from qeval import bench_evalplus as B
        items = B.load(tuple(a.subsets.split(",")))
    elif bench == "lcb":
        from qeval import bench_lcb as B
        items = B.load(tuple(a.releases.split(",")), set(a.difficulties.split(",")) if a.difficulties else None)
    elif bench == "bfcl":
        from qeval import bench_bfcl as B
        spec = {}
        for part in (a.bfcl_spec or DEFAULT_BFCL_SPEC).split(","):
            k, _, n = part.partition(":")
            spec[k.strip()] = int(n) if n.strip() and n.strip() != "all" else None
        items = B.load(spec)
    elif bench == "longctx":
        from qeval import bench_longctx as B
        items = B.load(set(a.bins.split(",")) if a.bins else None, None, a.per_bin)
    else:
        raise SystemExit(f"unknown bench {bench}")
    if a.limit:
        items = common.subsample(items, a.limit, seed=f"limit-{bench}")
    meta = {"bench": bench, "start": time.strftime("%Y-%m-%dT%H:%M:%S"), "cfg": cfg_public(c), "n_items": len(items),
            "repeats": a.repeats, "budget_min": a.budget_min, "argv": sys.argv[1:]}
    try:
        st, models = common.ChatClient(c).get_json("/models")
        meta["models"] = models
    except Exception as e:
        meta["models_error"] = str(e)
    runner = common.Runner(c, bench, out_path)
    t0 = time.time()
    runner.run(items, B.make_fn(c), concurrency=a.concurrency, budget_s=a.budget_min * 60 if a.budget_min else None,
               repeats=a.repeats, groups=(bench == "longctx" and not a.no_groups))
    meta["end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, f"{bench}{a.suffix}.meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    from qeval import report
    s = report.summarize_file(out_path)
    with open(os.path.join(out_dir, f"{bench}{a.suffix}.summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    log(f"[{bench}] summary: " + json.dumps({k: s["overall"][k] for k in ("n", "acc", "ci95", "runaway_rate")}))
    return 0


def cmd_rescore(a):
    """Re-run the code execution for stored evalplus/lcb results (no model calls)."""
    sandbox.setup(a.sandbox_python, exec_workers=a.exec_workers)
    recs = common.read_jsonl(a.results)
    bench = recs[0]["bench"]
    out = a.results.replace(".jsonl", ".rescored.jsonl")
    if bench == "evalplus":
        from qeval import bench_evalplus as B
        tasks = {it["id"]: it for it in B.load()}
    else:
        from qeval import bench_lcb as B
        tasks = {it["id"]: it for it in B.load(("v5", "v6"))}
    import gzip
    from concurrent.futures import ThreadPoolExecutor

    def one(r):
        code = r.get("code")
        if not code:
            return r
        it = tasks[r["id"]]
        if bench == "evalplus":
            t = it["task"]
            res = sandbox.run_worker("worker_evalplus.py", {
                "dataset": it["dataset"], "task_id": t["task_id"], "entry_point": t["entry_point"],
                "canonical": t["prompt"] + t["canonical_solution"], "solution": code, "base_input": t["base_input"],
                "plus_input": t["plus_input"], "atol": t.get("atol", 0)}, timeout=300)
            r["base_correct"] = res.get("base") == "pass"
            r["correct"] = r["base_correct"] and res.get("plus") == "pass"
        else:
            p = it["problem"]
            with gzip.open(os.path.join(common.DATA, "lcb", "tests_" + p.get("release", "v6"), p["question_id"] + ".json.gz"), "rt") as f:
                tests = json.load(f)
            res = sandbox.run_worker("worker_lcb.py", {"code": code, "tests": tests, "timeout": 6},
                                     timeout=min(900, 7 * len(tests["inputs"]) + 15))
            r["correct"] = bool(res.get("passed"))
        r["score"] = 1.0 if r["correct"] else 0.0
        r["rescored"] = True
        return r

    with ThreadPoolExecutor(a.exec_workers) as ex:
        new = list(ex.map(one, recs))
    with open(out, "w") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")
    log(f"rescored {len(new)} -> {out}: acc {sum(r['correct'] for r in new)}/{len(new)}")
    return 0


def cmd_compare(a):
    from qeval import report
    res = report.compare_dirs(a.dirs, a.labels.split(",") if a.labels else None, boot=a.bootstrap)
    os.makedirs(os.path.dirname(os.path.abspath(a.json_out)), exist_ok=True)
    with open(a.json_out, "w") as f:
        json.dump(res, f, indent=1)
    md = report.to_markdown(res)
    with open(a.json_out.replace(".json", ".md"), "w") as f:
        f.write(md)
    print(md)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("selftest", help="check the code sandbox (no model calls)")
    p.add_argument("--sandbox-python", default=None)
    p = sub.add_parser("probe", help="check endpoint, reasoning field, effort wiring, tool calls")
    add_common(p)
    p = sub.add_parser("run", help="run one benchmark")
    add_common(p)
    p.add_argument("--bench", required=True, choices=["evalplus", "lcb", "bfcl", "longctx"])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--budget-min", type=float, default=None, help="stop starting new items after N minutes")
    p.add_argument("--limit", type=int, default=None, help="deterministic subsample of N items")
    p.add_argument("--suffix", default="", help="results file suffix, e.g. _t06")
    p.add_argument("--subsets", default="humaneval,mbpp")
    p.add_argument("--releases", default="v6")
    p.add_argument("--difficulties", default=None)
    p.add_argument("--bfcl-spec", default=None, help="category:n,... (n=all for everything)")
    p.add_argument("--bins", default=None, help="e.g. 32k,64k,110k")
    p.add_argument("--per-bin", type=int, default=None, help="haystacks per bin (max 6)")
    p.add_argument("--no-groups", action="store_true")
    p = sub.add_parser("rescore", help="re-execute stored code (evalplus/lcb)")
    p.add_argument("results")
    p.add_argument("--exec-workers", type=int, default=3)
    p.add_argument("--sandbox-python", default=None)
    p = sub.add_parser("compare", help="paired comparison of result dirs")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--labels", default=None)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--json-out", default="results/compare.json")
    a = ap.parse_args()
    return {"selftest": cmd_selftest, "probe": cmd_probe, "run": cmd_run, "rescore": cmd_rescore,
            "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
