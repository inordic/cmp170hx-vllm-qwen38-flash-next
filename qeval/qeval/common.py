"""Shared plumbing for qeval: OpenAI-compatible streaming client, item runner, JSONL results.

Pure standard library (Python >= 3.10). Nothing here talks to anything except ENDPOINT.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import ssl
import sys
import threading
import time
import traceback
import urllib.parse
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.environ.get("QEVAL_DATA", os.path.join(ROOT, "data"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def stable_hash(*parts) -> int:
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


# --------------------------------------------------------------------------- client


class ChatClient:
    """Minimal OpenAI-compatible /v1/chat/completions client (streaming SSE, stdlib only)."""

    def __init__(self, cfg):
        self.cfg = cfg
        base = cfg.endpoint.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        u = urllib.parse.urlparse(base)
        self.scheme, self.host, self.port = u.scheme, u.hostname, u.port
        self.path = u.path + "/chat/completions"
        self.base_path = u.path
        self._ctx = None
        if self.scheme == "https":
            self._ctx = ssl.create_default_context()
            if cfg.insecure:
                self._ctx.check_hostname = False
                self._ctx.verify_mode = ssl.CERT_NONE

    def _conn(self, timeout):
        if self.scheme == "https":
            return http.client.HTTPSConnection(self.host, self.port or 443, timeout=timeout, context=self._ctx)
        return http.client.HTTPConnection(self.host, self.port or 80, timeout=timeout)

    def _headers(self):
        h = {"Content-Type": "application/json", "Accept": "text/event-stream, application/json"}
        if self.cfg.key:
            h["Authorization"] = f"Bearer {self.cfg.key}"
        return h

    def get_json(self, path, timeout=30):
        c = self._conn(timeout)
        try:
            c.request("GET", self.base_path + path, headers=self._headers())
            r = c.getresponse()
            body = r.read()
            return r.status, json.loads(body or b"{}")
        finally:
            c.close()

    def build_body(self, messages, tools=None, max_tokens=None, seed=None, extra=None):
        cfg = self.cfg
        body = {
            "model": cfg.model,
            "messages": messages,
            "max_tokens": int(max_tokens or cfg.max_tokens),
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "stream": bool(cfg.stream),
        }
        if cfg.top_k is not None and cfg.top_k >= 0:
            body["top_k"] = cfg.top_k
        if cfg.min_p is not None:
            body["min_p"] = cfg.min_p
        if cfg.presence_penalty:
            body["presence_penalty"] = cfg.presence_penalty
        if seed is not None and cfg.use_seed:
            body["seed"] = int(seed) & 0x7FFFFFFF
        if cfg.stream:
            body["stream_options"] = {"include_usage": True}
        if cfg.effort and cfg.effort.lower() != "none":
            if cfg.effort_mode in ("kwargs", "both"):
                body.setdefault("chat_template_kwargs", {})["reasoning_effort"] = cfg.effort
            if cfg.effort_mode in ("top", "both"):
                body["reasoning_effort"] = cfg.effort
        if tools:
            body["tools"] = tools
        if cfg.extra_body:
            deep_merge(body, cfg.extra_body)
        if extra:
            deep_merge(body, extra)
        return body

    def chat(self, messages, tools=None, max_tokens=None, seed=None, extra=None) -> dict:
        body = self.build_body(messages, tools, max_tokens, seed, extra)
        payload = json.dumps(body).encode()
        last_err = None
        for attempt in range(self.cfg.retries + 1):
            try:
                return self._once(payload, bool(body.get("stream")))
            except _Fatal as e:
                if "seed" in body and "seed" in str(e).lower():
                    # e.g. a server that rejects per-request seeds with speculative decoding
                    if self.cfg.use_seed:
                        self.cfg.use_seed = False
                        log(f"server rejected 'seed' ({str(e)[:200]}); continuing WITHOUT per-request seeds")
                    body.pop("seed", None)
                    payload = json.dumps(body).encode()
                    continue
                return _err_result(str(e))
            except Exception as e:  # transport problems: retry with backoff
                last_err = f"{type(e).__name__}: {e}"
                if attempt < self.cfg.retries:
                    time.sleep(min(60, 5 * (2 ** attempt)))
        return _err_result("transport: " + str(last_err), transport=True)

    def _once(self, payload, stream):
        t0 = time.perf_counter()
        c = self._conn(self.cfg.read_timeout)
        try:
            c.request("POST", self.path, body=payload, headers=self._headers())
            r = c.getresponse()
            if r.status != 200:
                txt = r.read()[:2000].decode("utf-8", "replace")
                if r.status in (408, 429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status}: {txt}")
                raise _Fatal(f"HTTP {r.status}: {txt}")
            if not stream:
                d = json.loads(r.read())
                t1 = time.perf_counter()
                ch = d["choices"][0]
                msg = ch.get("message") or {}
                res = _new_result()
                res["content"] = msg.get("content") or ""
                res["reasoning"] = msg.get("reasoning_content") or msg.get("reasoning") or ""
                for tc in msg.get("tool_calls") or []:
                    f = tc.get("function") or {}
                    res["tool_calls"].append({"id": tc.get("id") or "", "name": f.get("name") or "",
                                              "arguments": f.get("arguments") or ""})
                res["finish_reason"] = ch.get("finish_reason")
                res["usage"] = d.get("usage") or {}
                res["latency_s"] = t1 - t0
                return _finish(res)
            res = _new_result()
            content, reasoning, tcs = [], [], {}
            t_first = t_answer = None
            n_chunks = 0
            while True:
                line = r.readline()
                if not line:
                    break
                line = line.strip()
                if not line or not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    d = json.loads(data)
                except Exception:
                    continue
                if d.get("error"):
                    raise _Fatal(f"stream error: {json.dumps(d['error'])[:500]}")
                if d.get("usage"):
                    res["usage"] = d["usage"]
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    now = time.perf_counter()
                    rc = delta.get("reasoning_content")
                    if rc is None:
                        rc = delta.get("reasoning")
                    if rc:
                        reasoning.append(rc)
                        t_first = t_first or now
                    cc = delta.get("content")
                    if cc:
                        content.append(cc)
                        t_first = t_first or now
                        t_answer = t_answer or now
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        slot = tcs.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        f = tc.get("function") or {}
                        if f.get("name"):
                            slot["name"] += f["name"]
                        if f.get("arguments"):
                            slot["arguments"] += f["arguments"]
                        t_first = t_first or now
                        t_answer = t_answer or now
                    if ch.get("finish_reason"):
                        res["finish_reason"] = ch["finish_reason"]
                    n_chunks += 1
            t1 = time.perf_counter()
            res["content"] = "".join(content)
            res["reasoning"] = "".join(reasoning)
            res["tool_calls"] = [tcs[k] for k in sorted(tcs)]
            res["latency_s"] = t1 - t0
            res["ttft_s"] = (t_first - t0) if t_first else None
            res["t_answer_s"] = (t_answer - t0) if t_answer else None
            res["n_chunks"] = n_chunks
            if res["finish_reason"] is None:
                raise RuntimeError("stream ended without finish_reason")
            return _finish(res)
        finally:
            c.close()


class _Fatal(Exception):
    pass


def _new_result():
    return {"content": "", "reasoning": "", "tool_calls": [], "finish_reason": None, "usage": {},
            "latency_s": None, "ttft_s": None, "t_answer_s": None, "error": None}


def _err_result(msg, transport=False):
    r = _new_result()
    r["error"] = msg
    r["transport_error"] = transport
    return r


def _finish(res):
    # Servers without a reasoning parser put <think>...</think> into content.
    c = res["content"]
    if not res["reasoning"] and "</think>" in c:
        pre, _, post = c.rpartition("</think>")
        res["reasoning"] = pre.replace("<think>", "", 1)
        res["content"] = post
    elif not res["reasoning"] and c.lstrip().startswith("<think>"):
        res["reasoning"] = c.lstrip()[7:]
        res["content"] = ""
    return res


def deep_merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


# --------------------------------------------------------------------------- per-item context


def loop_ratio(text: str, window: int = 8000) -> float | None:
    """zlib ratio of the last `window` chars; < ~0.12 means the model is repeating itself."""
    t = text[-window:]
    if len(t) < 2000:
        return None
    b = t.encode("utf-8", "replace")
    return round(len(zlib.compress(b, 6)) / len(b), 4)


class ItemCtx:
    """Collects the model calls made for one item (one or many, e.g. BFCL multi-turn)."""

    def __init__(self, client: ChatClient, cfg, seed: int):
        self.client, self.cfg, self.seed = client, cfg, seed
        self.calls = []

    def chat(self, messages, tools=None, max_tokens=None, extra=None):
        seed = self.seed + len(self.calls)
        r = self.client.chat(messages, tools=tools, max_tokens=max_tokens, seed=seed, extra=extra)
        self.calls.append(r)
        return r

    def summary(self):
        u_p = sum((c.get("usage") or {}).get("prompt_tokens") or 0 for c in self.calls)
        u_c = sum((c.get("usage") or {}).get("completion_tokens") or 0 for c in self.calls)
        rch = sum(len(c["reasoning"]) for c in self.calls)
        cch = sum(len(c["content"]) + sum(len(t["arguments"]) for t in c["tool_calls"]) for c in self.calls)
        fins = [c.get("finish_reason") for c in self.calls]
        runaway = any(f == "length" for f in fins)
        first = self.calls[0] if self.calls else {}
        last = self.calls[-1] if self.calls else {}
        lr = loop_ratio(last.get("reasoning", "") + last.get("content", "")) if last else None
        return {
            "n_calls": len(self.calls),
            "prompt_tokens": u_p,
            "completion_tokens": u_c,
            # vLLM does not split reasoning/answer tokens; apportion by characters
            "est_reasoning_tokens": int(round(u_c * rch / max(1, rch + cch))) if u_c else None,
            "reasoning_chars": rch,
            "answer_chars": cch,
            "max_call_completion_tokens": max([(c.get("usage") or {}).get("completion_tokens") or 0
                                               for c in self.calls] or [0]),
            "finish_reasons": fins,
            "runaway": runaway,
            "runaway_no_answer": runaway and any(c.get("finish_reason") == "length" and not c["content"]
                                                 and not c["tool_calls"] for c in self.calls),
            "loop_ratio": lr,
            "looping": (lr is not None and lr < 0.12 and runaway),
            "ttft_s": first.get("ttft_s"),
            "t_answer_s": first.get("t_answer_s") if len(self.calls) == 1 else None,
            "latency_s": round(sum(c.get("latency_s") or 0 for c in self.calls), 3),
            "api_error": next((c["error"] for c in self.calls if c.get("error")), None),
            "transport_error": any(c.get("transport_error") for c in self.calls),
        }


# --------------------------------------------------------------------------- runner


class Runner:
    """Runs items with a thread pool, resumes from an existing JSONL, honours a time budget.

    Items are dicts with at least "id". `fn(item, ctx) -> dict` must return a dict with
    "correct" (bool) or "score" (float 0..1) plus any extra fields worth keeping.
    """

    def __init__(self, cfg, bench, out_path):
        self.cfg, self.bench, self.out_path = cfg, bench, out_path
        self.client = ChatClient(cfg)
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    def _done(self):
        done = {}
        if os.path.exists(self.out_path):
            with open(self.out_path) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    if d.get("transport_error"):
                        continue  # retry those on resume
                    done[(d["id"], d.get("repeat", 0))] = d
        return done

    def _write(self, rec):
        line = json.dumps(rec, ensure_ascii=False)
        with self.lock:
            with open(self.out_path, "a") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

    def run(self, items, fn, concurrency=None, budget_s=None, repeats=1, order="shuffle", groups=False):
        cfg = self.cfg
        concurrency = concurrency or cfg.concurrency
        done = self._done()
        work = []
        for r in range(repeats):
            for it in items:
                if (it["id"], r) not in done:
                    work.append((it, r))
        if order == "shuffle":
            # same order for every model: shuffle by a hash of the id, repeats after originals
            work.sort(key=lambda x: (x[1], stable_hash(cfg.order_seed, self.bench, x[0]["id"])))
        log(f"[{self.bench}] {len(items)} items x {repeats} repeats; {len(done)} already done; "
            f"{len(work)} to run; concurrency={concurrency}; budget={f'{budget_s:.0f}s' if budget_s else 'none'}")
        t_start = time.time()
        deadline = t_start + budget_s if budget_s else None
        stats = {"ok": 0, "correct": 0, "runaway": 0, "err": 0, "consec_transport": 0}
        self.abort = False

        def task(it, r):
            seed = cfg.seed_base + stable_hash(self.bench, it["id"], r) % 1_000_000
            ctx = ItemCtx(self.client, cfg, seed)
            t0 = time.time()
            try:
                out = fn(it, ctx) or {}
            except Exception as e:
                out = {"correct": False, "harness_error": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()[-2000:]}
            rec = {"bench": self.bench, "id": it["id"], "repeat": r, "seed": seed,
                   "subset": it.get("subset"), "group": it.get("group")}
            rec.update(ctx.summary())
            rec.update(out)
            if "score" not in rec:
                rec["score"] = 1.0 if rec.get("correct") else 0.0
            rec["wall_s"] = round(time.time() - t0, 3)
            rec["model_tag"] = cfg.tag
            rec["effort"] = cfg.effort
            self._write(rec)
            with self.lock:
                stats["ok"] += 1
                stats["correct"] += 1 if rec.get("correct") else 0
                stats["runaway"] += 1 if rec.get("runaway") else 0
                stats["err"] += 1 if (rec.get("api_error") or rec.get("harness_error")) else 0
                stats["consec_transport"] = stats["consec_transport"] + 1 if rec.get("transport_error") else 0
                if stats["consec_transport"] >= 2 * concurrency and not self.abort:
                    self.abort = True
                    log(f"[{self.bench}] {stats['consec_transport']} transport errors in a row - server down? "
                        f"stopping this phase (re-run the same command to resume)")
                n = stats["ok"]
                if n % max(1, min(25, len(work) // 10 or 1)) == 0 or n == len(work):
                    log(f"[{self.bench}] {n}/{len(work)} done, acc so far {stats['correct']}/{n}, "
                        f"runaway {stats['runaway']}, errors {stats['err']}, {time.time() - t_start:.0f}s")
            return rec

        skipped = 0
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            if groups:
                self._run_grouped(ex, work, task, concurrency, deadline)
            else:
                pending = set()
                it_work = iter(work)
                exhausted = False
                while True:
                    while not exhausted and len(pending) < concurrency:
                        if self.abort:
                            exhausted = True
                            break
                        if deadline and time.time() > deadline:
                            exhausted = True
                            skipped = len(work) - stats["ok"] - len(pending)
                            break
                        try:
                            it, r = next(it_work)
                        except StopIteration:
                            exhausted = True
                            break
                        pending.add(ex.submit(task, it, r))
                    if not pending:
                        break
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in finished:
                        f.result()
        if skipped:
            log(f"[{self.bench}] time budget reached: {skipped} items not started")
        return list(self._done().values())

    def _run_grouped(self, ex, work, task, concurrency, deadline):
        """Long-context mode: per group (shared haystack), run one item first to warm the prefix
        cache, then release the rest of that group. Groups are pipelined."""
        by_group = {}
        for it, r in work:
            by_group.setdefault((it.get("group"), r), []).append((it, r))
        sem = threading.Semaphore(concurrency)

        def run_one(it, r):
            try:
                return task(it, r)
            finally:
                sem.release()

        futs = []
        for key in sorted(by_group, key=lambda k: (k[1], str(k[0]))):
            if self.abort:
                break
            if deadline and time.time() > deadline:
                log(f"[{self.bench}] time budget reached before group {key}")
                break
            lst = by_group[key]
            sem.acquire()
            first = ex.submit(run_one, *lst[0])
            first.result()
            for it, r in lst[1:]:
                if self.abort:
                    break
                sem.acquire()
                futs.append(ex.submit(run_one, it, r))
        for f in futs:
            f.result()


# --------------------------------------------------------------------------- answer helpers


CODE_BLOCK_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+\-]*)[^\n]*\n(.*?)```", re.S)


def extract_code(text: str, prefer_lang="python", must_contain=None) -> str:
    """Last fenced python block (or the last block containing `must_contain`)."""
    blocks = CODE_BLOCK_RE.findall(text or "")
    if not blocks:
        # unterminated final block
        m = re.search(r"```[ \t]*(?:python|py)?[^\n]*\n(.*)$", text or "", re.S)
        return m.group(1) if m else ""
    py = [b for lang, b in blocks if lang.lower() in ("python", "py", "python3", "")]
    cand = py or [b for _, b in blocks]
    if must_contain:
        with_name = [b for b in cand if must_contain in b]
        if with_name:
            return with_name[-1]
    return cand[-1]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def read_jsonl(path):
    out = []
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def subsample(items, n, seed, key=lambda x: x["id"]):
    if n is None or n <= 0 or n >= len(items):
        return list(items)
    s = sorted(items, key=lambda x: stable_hash("subsample", seed, key(x)))
    return sorted(s[:n], key=key)
