#!/usr/bin/env python3
"""Mock OpenAI-compatible server for testing the qeval harness end to end (NOT a model).

It answers with ground truth where it can (EvalPlus canonical solutions, BFCL possible answers and
multi-turn ground truth, long-context answers), streams a fake reasoning field first, and makes
every 25th request a "runaway" (finish_reason=length). Scores should therefore be ~100% except
LCB (no reference solutions: it returns a stub) and the injected runaways.

    python3 tools/mock_server.py --port 18080
"""
import argparse
import ast
import hashlib
import inspect
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qeval import bench_bfcl, bench_evalplus, bench_longctx  # noqa: E402
from qeval.bfcl_vendor import multi_turn as MT  # noqa: E402

LOCK = threading.Lock()
COUNT = [0]
EP, BF1, BFM, LC = {}, {}, {}, {}


def load_tables():
    for it in bench_evalplus.load():
        t = it["task"]
        EP[t["prompt"].strip()] = t["prompt"] + t["canonical_solution"]
    spec = {c: None for c in bench_bfcl.SINGLE_TURN + bench_bfcl.MULTI_TURN}
    for it in bench_bfcl.load(spec):
        e = it["entry"]
        if it["category"].startswith("multi_turn"):
            BFM[e["question"][0][0]["content"]] = it
        else:
            BF1[json.dumps(e["question"][0], sort_keys=True)] = it
    for it in bench_longctx.load():
        LC[it["question"]] = it


def _signature_order(it):
    ex = MT.Executor(it["entry"].get("initial_config", {}), it["entry"]["involved_classes"])
    return {k: list(inspect.signature(m).parameters) for k, m in ex.namespace.items()}


def parse_call(s, sig):
    node = ast.parse(s, mode="eval").body
    name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
    args = {}
    names = sig.get(name, [])
    for i, a in enumerate(node.args):
        args[names[i]] = ast.literal_eval(a)
    for kw in node.keywords:
        args[kw.arg] = ast.literal_eval(kw.value)
    return name, args


def answer(body):
    msgs = body["messages"]
    tools = body.get("tools")
    last_user = [m for m in msgs if m["role"] == "user"]
    text, calls = "I could not find this.", []
    if tools:
        first_user = last_user[0]["content"] if last_user else ""
        key = json.dumps([m for m in msgs], sort_keys=True)
        if key in BF1:
            it = BF1[key]
            cat = it["category"]
            if "irrelevance" in cat:
                text = "None of the functions fit this request."
            elif cat == "live_relevance":
                f = it["entry"]["function"][0]
                calls = [(f["name"].replace(".", "_"), {})]
            else:
                for fc in it["answer"]:
                    for fname, params in fc.items():
                        args = {}
                        for p, opts in params.items():
                            vals = [o for o in opts if o != ""]
                            if vals:
                                args[p] = vals[0]
                        calls.append((fname.replace(".", "_"), args))
        elif first_user in BFM:
            it = BFM[first_user]
            if msgs[-1]["role"] == "tool":
                text = "Done."
            else:
                turn = len(last_user) - 1
                sig = _signature_order(it)
                gt = it["answer"][turn] if turn < len(it["answer"]) else []
                calls = [parse_call(s, sig) for s in gt]
                if not calls:
                    text = "I cannot do that with the available functions."
    else:
        u = last_user[-1]["content"] if last_user else ""
        if u.startswith("Please provide a self-contained Python script"):
            pr = u.split("```python\n", 1)[1].rsplit("\n```", 1)[0].strip()
            sol = EP.get(pr, "def f():\n    pass\n")
            text = f"Here is the solution:\n```python\n{sol}\n```"
        elif "</codebase>" in u:
            q = u.split("</codebase>\n\n", 1)[1]
            it = LC.get(q)
            if it:
                a = it["answer"]
                a = ", ".join(str(x) for x in a) if isinstance(a, list) else str(a)
                text = f"Found it.\nANSWER: {a}"
        elif "### Question:" in u:
            text = "```python\nimport sys\nprint(sys.stdin.read().strip())\n```"
        elif "prime numbers" in u:
            text = "17"
        elif "weather" in u:
            text = "ok"
    return text, calls


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.endswith("/models"):
            return self._send(200, {"object": "list", "data": [{"id": "mock-model", "root": "/mock"}]})
        self._send(404, {"error": "nf"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        with LOCK:
            COUNT[0] += 1
            idx = COUNT[0]
        effort = (body.get("chat_template_kwargs") or {}).get("reasoning_effort", "medium")
        text, calls = answer(body)
        runaway = idx % 25 == 0 and not body.get("tools")
        reasoning = {"low": "Short think. ", "medium": "Let me think about it. " * 5, "high": "Deep thought. " * 40}.get(effort, "x ")
        rid = hashlib.md5(str(idx).encode()).hexdigest()[:12]
        if not body.get("stream"):
            msg = {"role": "assistant", "content": text if not calls else None, "reasoning": reasoning}
            if calls:
                msg["tool_calls"] = [{"id": f"call_{rid}_{i}", "type": "function", "function": {"name": c, "arguments": json.dumps(a)}}
                                     for i, (c, a) in enumerate(calls)]
            return self._send(200, {"choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
                                    "usage": {"prompt_tokens": 100, "completion_tokens": 50 + len(reasoning) // 4}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(obj):
            data = b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n"
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def ch(delta, fin=None):
            return {"id": rid, "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}

        for part in [reasoning[i:i + 40] for i in range(0, len(reasoning), 40)]:
            chunk(ch({"reasoning": part}))
            time.sleep(0.001)
        if runaway:
            chunk(ch({"reasoning": "and again " * 50}, "length"))
            ct = body.get("max_tokens", 100)
        else:
            if calls:
                for i, (c, a) in enumerate(calls):
                    chunk(ch({"tool_calls": [{"index": i, "id": f"call_{rid}_{i}", "type": "function",
                                              "function": {"name": c, "arguments": ""}}]}))
                    s = json.dumps(a)
                    for j in range(0, len(s), 20):
                        chunk(ch({"tool_calls": [{"index": i, "function": {"arguments": s[j:j + 20]}}]}))
                chunk(ch({}, "tool_calls"))
            else:
                for j in range(0, len(text), 30):
                    chunk(ch({"content": text[j:j + 30]}))
                chunk(ch({}, "stop"))
            ct = 50 + len(reasoning) // 4 + len(text) // 4
        chunk({"id": rid, "choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": ct}})
        chunk(b"[DONE]")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18080)
    a = ap.parse_args()
    load_tables()
    print(f"mock server on 127.0.0.1:{a.port} (evalplus {len(EP)}, bfcl {len(BF1)}+{len(BFM)}, longctx {len(LC)})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
