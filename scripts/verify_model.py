#!/usr/bin/env python3
"""Check every file of a downloaded Hugging Face snapshot against the hub: size for all files, sha256 for LFS files.

usage: verify_model.py REPO REVISION LOCAL_DIR [THREADS]
Prints one line per bad/missing file and a summary; exit code 1 if anything is wrong.
"""
import hashlib, json, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

repo, rev, local = sys.argv[1], sys.argv[2], Path(sys.argv[3])
threads = int(sys.argv[4]) if len(sys.argv) > 4 else 6
url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true&expand=true"
entries = [e for e in json.load(urllib.request.urlopen(url, timeout=60)) if e.get("type") == "file"]


def check(e: dict) -> str | None:
    p = local / e["path"]
    if not p.is_file():
        return f"MISSING {e['path']}"
    size = (e.get("lfs") or {}).get("size", e.get("size"))
    if p.stat().st_size != size:
        return f"SIZE    {e['path']}: {p.stat().st_size} != {size}"
    oid = (e.get("lfs") or {}).get("oid")
    if oid:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(16 << 20):
                h.update(chunk)
        if h.hexdigest() != oid:
            return f"SHA256  {e['path']}"
    return None


with ThreadPoolExecutor(threads) as ex:
    bad = [r for r in ex.map(check, entries) if r]
for line in bad:
    print(line)
total = sum((e.get("lfs") or {}).get("size", e.get("size", 0)) for e in entries)
print(f"{len(entries)} files, {total / 1e9:.1f} GB, {len(bad)} bad")
sys.exit(1 if bad else 0)
