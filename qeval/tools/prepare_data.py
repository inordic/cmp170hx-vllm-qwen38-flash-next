#!/usr/bin/env python3
"""Build quality/data/ from the raw downloads in quality/dl/ (run on the Mac, once).

    python3 -I tools/prepare_data.py --dl <quality>/dl --out <quality>/data [--tokenizer tokenizer.json]

- evalplus: copies the two release .jsonl.gz files (HumanEval+ v0.1.10, MBPP+ v0.2.0).
- lcb:      converts code_generation_lite test5/test6.jsonl -> problems_v5/v6.jsonl + tests/<id>.json.gz.
            Hidden tests are base64+zlib+pickle upstream; they are decoded here with an Unpickler
            that refuses every global, so only plain str payloads can load (no code execution),
            and are stored as plain JSON. The server never unpickles anything.
- bfcl:     copies BFCL v4 json files (questions, possible answers, multi-turn function docs).
- longctx:  generates the synthetic long-context set (code haystacks from the Python stdlib
            + inserted needles), token-counted with the Qwen3.8 tokenizer when available.
"""
import argparse
import base64
import gzip
import io
import json
import os
import pickle
import random
import shutil
import sys
import zlib


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):  # noqa: D401
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")


def safe_unpickle_str(b: bytes) -> str:
    obj = _NoGlobals(io.BytesIO(b)).load()
    if not isinstance(obj, str):
        raise ValueError(f"unexpected pickle payload type {type(obj)}")
    return obj


def prep_evalplus(dl, out):
    d = os.path.join(out, "evalplus")
    os.makedirs(d, exist_ok=True)
    for src, dst in (("HumanEvalPlus.jsonl.gz", "HumanEvalPlus-v0.1.10.jsonl.gz"),
                     ("MbppPlus.jsonl.gz", "MbppPlus-v0.2.0.jsonl.gz")):
        shutil.copyfile(os.path.join(dl, "evalplus", src), os.path.join(d, dst))
        with gzip.open(os.path.join(d, dst), "rt") as f:
            n = sum(1 for _ in f)
        print(f"evalplus {dst}: {n} tasks")


def prep_lcb(dl, out):
    d = os.path.join(out, "lcb")
    td = None
    os.makedirs(d, exist_ok=True)
    for rel, fname in (("v5", "test5.jsonl"), ("v6", "test6.jsonl")):
        src = os.path.join(dl, "lcb", fname)
        if not os.path.exists(src):
            print(f"lcb: {src} missing, skipped")
            continue
        td = os.path.join(d, f"tests_{rel}")
        os.makedirs(td, exist_ok=True)
        n = 0
        tot_tests = 0
        with open(src) as f, open(os.path.join(d, f"problems_{rel}.jsonl"), "w") as fo:
            for line in f:
                p = json.loads(line)
                pub = json.loads(p["public_test_cases"])
                raw = p["private_test_cases"]
                try:
                    priv = json.loads(raw)
                except Exception:
                    priv = json.loads(safe_unpickle_str(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))
                meta = json.loads(p.get("metadata") or "{}")
                tests = {"inputs": [t["input"] for t in pub + priv], "outputs": [t["output"] for t in pub + priv],
                         "fn_name": meta.get("func_name")}
                with gzip.open(os.path.join(td, p["question_id"] + ".json.gz"), "wt") as tf:
                    json.dump(tests, tf)
                tot_tests += len(tests["inputs"])
                rec = {k: p[k] for k in ("question_id", "question_title", "question_content", "platform",
                                         "contest_id", "contest_date", "starter_code", "difficulty")}
                rec["fn_name"] = meta.get("func_name")
                rec["n_tests"] = len(tests["inputs"])
                rec["release"] = rel
                fo.write(json.dumps(rec) + "\n")
                n += 1
        print(f"lcb {rel}: {n} problems, {tot_tests} tests")


def prep_bfcl(dl, out):
    src = os.path.join(dl, "bfcl", "gorilla", "berkeley-function-call-leaderboard", "bfcl_eval", "data")
    d = os.path.join(out, "bfcl")
    os.makedirs(os.path.join(d, "possible_answer"), exist_ok=True)
    cats = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance", "live_simple",
            "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance", "live_relevance",
            "multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"]
    for c in cats:
        f = f"BFCL_v4_{c}.json"
        shutil.copyfile(os.path.join(src, f), os.path.join(d, f))
        pa = os.path.join(src, "possible_answer", f)
        if os.path.exists(pa):
            shutil.copyfile(pa, os.path.join(d, "possible_answer", f))
        with open(os.path.join(d, f)) as fh:
            n = sum(1 for l in fh if l.strip())
        print(f"bfcl {c}: {n}")
    if os.path.exists(os.path.join(d, "multi_turn_func_doc")):
        shutil.rmtree(os.path.join(d, "multi_turn_func_doc"))
    shutil.copytree(os.path.join(src, "multi_turn_func_doc"), os.path.join(d, "multi_turn_func_doc"))
    shutil.copyfile(os.path.join(dl, "bfcl", "gorilla", "LICENSE"), os.path.join(d, "LICENSE"))


# ---------------------------------------------------------------- long context

WORDS = """amber azure basil birch bison cedar cobalt coral crane delta ember falcon fennel fjord garnet
glacier granite harbor hazel heron indigo iris jasper juniper kestrel lagoon larch lotus lynx maple
marble meadow merlin mesa nectar nimbus oasis obsidian ochre onyx orchid osprey otter pebble pepper
pine plume poplar quartz quill raven reef ridge rowan saffron sage sable sequoia sierra sorrel spruce
summit tamarind teal thistle tundra umber valley vesper violet walnut willow wren yarrow zephyr zinc
anchor beacon canyon dune echo flint grove hollow isle jade kelp lumen moss nova opal prism rapids
slate tide umbra vapor wharf yew zenith alder brook cinder dawn fern gale haven ivory jet kiln""".split()

STDLIB_SKIP = ("test", "tests", "idlelib", "tkinter", "turtledemo", "lib2to3", "site-packages", "ensurepip",
               "pydoc_data", "config-3", "__pycache__", "encodings", "unittest/test", "ctypes/test")


def stdlib_files(libdir):
    files = []
    for dp, dn, fn in os.walk(libdir):
        rel = os.path.relpath(dp, libdir)
        if any(rel == s or rel.startswith(s + "/") or ("/" + s + "/") in ("/" + rel + "/") for s in STDLIB_SKIP):
            continue
        for f in fn:
            if f.endswith(".py"):
                p = os.path.join(dp, f)
                try:
                    txt = open(p, encoding="utf-8").read()
                except Exception:
                    continue
                if 2000 < len(txt) < 120000:
                    files.append((os.path.relpath(p, libdir), txt))
    files.sort()
    return files


class Counter:
    def __init__(self, tok_path):
        self.tok = None
        if tok_path:
            try:
                from tokenizers import Tokenizer
                self.tok = Tokenizer.from_file(tok_path)
            except Exception as e:
                print("tokenizer unavailable, using chars/3.3 estimate:", e)

    def count(self, s):
        if self.tok is None:
            return int(len(s) / 3.3)
        return len(self.tok.encode(s, add_special_tokens=False).ids)


PREAMBLE = ("Below is a dump of a large Python code base (several modules concatenated). Read it carefully; "
            "a question about it follows after the dump.\n\n<codebase>\n")
POST = "\n</codebase>\n\n"
SUFFIX = ("\n\nAnswer from the code base above only. End your reply with a final line of the form "
          "`ANSWER: <answer>`.")


def _toplevel_starts(txt):
    """0-based line indices where a top-level statement starts (decorators included)."""
    import ast as _ast
    try:
        tree = _ast.parse(txt)
    except SyntaxError:
        return []
    out = []
    for node in tree.body:
        ln = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
        out.append(ln - 1)
    return out


def make_haystack(files, target_tokens, rnd, counter, reserve):
    """Returns (lines, insertion_points). Insertion points are line indices before which a
    module-level statement can be inserted (file boundaries and top-level statement starts)."""
    order = list(range(len(files)))
    rnd.shuffle(order)
    lines, pts = [], []
    tot = counter.count(PREAMBLE + POST) + reserve
    for idx in order:
        rel, txt = files[idx]
        body = txt.rstrip().split("\n")
        block = f"# ===== file: {rel} =====\n" + "\n".join(body) + "\n\n"
        c = counter.count(block)
        if tot + c > target_tokens:
            continue
        starts = _toplevel_starts(txt)
        if not starts:
            continue
        base = len(lines)
        pts.append(base)  # before the file header
        for st in starts:
            if st > 0:
                pts.append(base + 1 + st)
        lines.append(f"# ===== file: {rel} =====")
        lines.extend(body)
        lines.append("")
        tot += c
        if tot > target_tokens - 1500:
            break
    return lines, pts


def gen_longctx(out, tok_path, libdir, bins, per_bin, seed=20261014):
    counter = Counter(tok_path)
    files = stdlib_files(libdir)
    print(f"longctx: {len(files)} stdlib files from {libdir}")
    d = os.path.join(out, "longctx")
    hd = os.path.join(d, "haystacks")
    if os.path.exists(hd):
        shutil.rmtree(hd)
    os.makedirs(hd, exist_ok=True)
    items = []
    rnd = random.Random(seed)
    for target in bins:
        for h in range(per_bin):
            hid = f"L{target // 1024}k_h{h}"
            # leave room for needles (~1.2K tokens) and the longest question
            lines, pts = make_haystack(files, target, rnd, counter, reserve=1600)
            hay = "\n".join(lines)
            used_words = set()

            def name(prefix, k=2):
                while True:
                    ws = rnd.sample(WORDS, k)
                    n = prefix + "_".join(w.upper() for w in ws)
                    if n not in used_words and n not in hay:
                        used_words.add(n)
                        return n, ws

            def val():
                return rnd.randint(100000, 9999999)

            inserts = []  # (fraction, text)
            questions = []
            # A) 12 multi-key needles, evenly spread; ask 4
            kv = []
            first_word = rnd.choice(WORDS).upper()
            for i in range(12):
                frac = (i + rnd.uniform(0.15, 0.85)) / 12
                # half of the keys share a first word to make near-duplicate distractors
                if i % 2 == 0:
                    while True:
                        w2 = rnd.choice(WORDS).upper()
                        key = f"CFG_{first_word}_{w2}"
                        if key not in used_words and key not in hay:
                            used_words.add(key)
                            break
                else:
                    key, _ = name("CFG_")
                v = val()
                kv.append((key, v, frac))
                inserts.append((frac, f"{key} = {v}  # deployment constant\n"))
            for j in rnd.sample(range(12), 4):
                key, v, frac = kv[j]
                questions.append({"kind": "kv", "q": f"What integer value is assigned to the module-level constant `{key}`?",
                                  "answer": v, "depth": round(frac, 3)})
            # B) multi-value: 2 route names, 3 registrations each; 6 single-value distractors
            for m in range(2):
                route = "_".join(rnd.sample(WORDS, 2))
                vals = []
                fr = sorted(rnd.uniform(0.02, 0.98) for _ in range(3))
                for f_ in fr:
                    port = rnd.randint(20000, 65000)
                    vals.append(port)
                    inserts.append((f_, f'register_route("{route}", {port})\n'))
                questions.append({"kind": "multivalue",
                                  "q": f'The code calls `register_route("{route}", <port>)` more than once. List every port number registered for the route "{route}", comma-separated.',
                                  "answer": vals, "depth": round(sum(fr) / 3, 3)})
            for _ in range(6):
                route = "_".join(rnd.sample(WORDS, 2))
                inserts.append((rnd.uniform(0.02, 0.98), f'register_route("{route}", {rnd.randint(20000, 65000)})\n'))
            # C) variable tracking: 2 chains of 4 names, plus one 2-name distractor chain each
            for m in range(2):
                v = val()
                names = [name("TOKEN_")[0] for _ in range(4)]
                fr = sorted(rnd.uniform(0.02, 0.98) for _ in range(4))
                inserts.append((fr[0], f"{names[0]} = {v}\n"))
                for k in range(1, 4):
                    inserts.append((fr[k], f"{names[k]} = {names[k - 1]}\n"))
                dn = [name("TOKEN_")[0] for _ in range(2)]
                dv = v + rnd.choice([1, -1]) * rnd.randint(1, 9) * 10
                df = sorted(rnd.uniform(0.02, 0.98) for _ in range(2))
                inserts.append((df[0], f"{dn[0]} = {dv}\n"))
                inserts.append((df[1], f"{dn[1]} = {dn[0]}\n"))
                questions.append({"kind": "vartrack",
                                  "q": f"Which module-level names named TOKEN_* end up holding the integer {v}, either directly or through assignment from another TOKEN_* name? List all of them, comma-separated.",
                                  "answer": names, "depth": round(sum(fr) / 4, 3)})
            # D) function + far-away constant
            for m in range(2):
                fname = "quota_for_" + "_".join(rnd.sample(WORDS, 2))
                cname, _ = name("SCALE_")
                a, b, c = rnd.randint(3, 19), rnd.randint(10, 999), rnd.randint(2, 40)
                f1, f2 = rnd.uniform(0.02, 0.98), rnd.uniform(0.02, 0.98)
                inserts.append((f1, f"def {fname}(n):\n    return n * {cname} + {b}\n"))
                inserts.append((f2, f"{cname} = {a}\n"))
                questions.append({"kind": "func", "q": f"What does `{fname}({c})` return? Give the integer.",
                                  "answer": c * a + b, "depth": round((f1 + f2) / 2, 3)})
            # place inserts at legal top-level insertion points (stable order)
            placed = {}
            for frac, text in sorted(inserts, key=lambda x: x[0]):
                i = pts[min(len(pts) - 1, int(frac * len(pts)))]
                placed.setdefault(i, []).append(text)
            new_lines = []
            for i, line in enumerate(lines):
                if i in placed:
                    for t in placed[i]:
                        new_lines.append(t.rstrip("\n"))
                    new_lines.append("")
                new_lines.append(line)
            hay2 = "\n".join(new_lines)
            prefix = PREAMBLE + hay2 + POST
            with open(os.path.join(hd, hid + ".txt"), "w") as f:
                f.write(prefix)
            ptoks = counter.count(prefix)
            for qi, q in enumerate(questions):
                full = "Question: " + q["q"] + SUFFIX
                items.append({"id": f"{hid}_q{qi}", "group": hid, "bin": f"{target // 1024}k", "kind": q["kind"],
                              "question": full, "answer": q["answer"], "depth": q["depth"],
                              "prompt_tokens_est": ptoks + counter.count(full)})
            print(f"longctx {hid}: {ptoks} tokens (target {target}), {len(questions)} questions")
    with open(os.path.join(d, "items.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    with open(os.path.join(d, "SOURCE.txt"), "w") as f:
        f.write(f"Haystack = Python {sys.version.split()[0]} standard library sources from {libdir} "
                f"(PSF License), concatenated; needles are synthetic. tokenizer={tok_path} seed={seed}\n")
    print(f"longctx: {len(items)} items")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="evalplus,lcb,bfcl,longctx")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--stdlib", default=None)
    ap.add_argument("--bins", default="32768,65536,112640")
    ap.add_argument("--per-bin", type=int, default=6)
    a = ap.parse_args()
    only = set(a.only.split(","))
    os.makedirs(a.out, exist_ok=True)
    if "evalplus" in only:
        prep_evalplus(a.dl, a.out)
    if "lcb" in only:
        prep_lcb(a.dl, a.out)
    if "bfcl" in only:
        prep_bfcl(a.dl, a.out)
    if "longctx" in only:
        lib = a.stdlib or os.path.dirname(os.__file__)
        gen_longctx(a.out, a.tokenizer, lib, [int(x) for x in a.bins.split(",")], a.per_bin)


if __name__ == "__main__":
    main()
