#!/bin/bash
# Night 4: the same speed + correctness set for every config (server on 8008). usage: [QUICK=1] speed.sh NAME
# QUICK=1: decode tests only (code + code edit), for restart-to-restart comparisons.
# PF=1: prefill sweep (8K/32K/128K prefill + code decode), no edit/func/multiturn.
# Writes $N/res-NAME.json (machine-readable, used by select.py) and lines in NIGHT-LOG.md.
set -uo pipefail
NAME=$1; N=/root/tune/night4; NL=$N/NIGHT-LOG.md; T=/root/tune
VPY=/opt/vllm-qfn-exp/.venv/bin/python; TOK=/opt/vllm-qfn/models/swift15-awq
note() { echo "- $(date -u +%T) UTC [$NAME] $*" >> $NL; }
cd $T
QUICK=${QUICK:-0}; PF=${PF:-0}
[ "$PF" = 1 ] && PERF_SIZES=${PERF_SIZES:-8192,32768,131072}
[ "$QUICK" = 1 ] || PERF_SIZES=${PERF_SIZES:-1024,8192,32768} $VPY perf_exact.py n4-$NAME 8008 $TOK 3 256 > $N/perf-$NAME.txt 2>&1
python3 -I decode_code.py n4-$NAME 8008 > $N/code-$NAME.txt 2>&1
if [ "$PF" = 1 ]; then : > $N/edit-$NAME.txt; else python3 -I edit_decode.py n4-$NAME 8008 > $N/edit-$NAME.txt 2>&1; fi
if [ "$QUICK" = 1 ] || [ "$PF" = 1 ]; then : > $N/func-$NAME.txt; : > $N/multi-$NAME.txt
else
  python3 -I func_check.py 8008 > $N/func-$NAME.txt 2>&1
  python3 -I multiturn_check.py 8008 3 20000 > $N/multi-$NAME.txt 2>&1
fi
$VPY - "$NAME" <<'EOF' >> $NL
import json, re, statistics as st, sys
name = sys.argv[1]; N = "/root/tune/night4"; r = {"name": name}
try:
    rows = json.load(open(f"/root/tune/exact-n4-{name}.json")); by = {}
    for x in rows: by.setdefault(x["size"], []).append(x)
    r["pp"] = {k: st.median(y["pp"] for y in v) for k, v in by.items()}
    r["tg"] = {k: st.median(y["tg"] for y in v) for k, v in by.items()}
except Exception as e: r["perf_error"] = str(e)
try:
    d = json.load(open(f"/root/tune/decode-n4-{name}.json"))
    r["code"] = st.median(x["tg"] for x in d["rows"]); r["accept"] = d["acceptance_length"]
except Exception as e: r["code_error"] = str(e)
for t in open(f"{N}/edit-{name}.txt").read().splitlines():
    m = re.search(r"thinking (off|low) \| decode tok/s median ([\d.]+)", t)
    if m: r["edit_" + m.group(1)] = float(m.group(2))
r["func_ok"] = "FUNC ALL PASS" in open(f"{N}/func-{name}.txt").read()
r["multi_ok"] = "MULTITURN ALL PASS" in open(f"{N}/multi-{name}.txt").read()
json.dump(r, open(f"{N}/res-{name}.json", "w"), indent=1)
f = lambda d: " ".join(f"{k // 1024}K {v:.0f}" for k, v in sorted(d.items())) if d else "-"
print(f"- [{name}] prefill {f(r.get('pp', {}))} | decode-at-size {f(r.get('tg', {}))} | code {r.get('code', 0):.1f} tok/s "
      f"(accept {r.get('accept', 0):.2f}) | edit off {r.get('edit_off', 0):.1f} / low {r.get('edit_low', 0):.1f} | "
      + (f"func {'PASS' if r['func_ok'] else 'FAIL'} | multiturn {'PASS' if r['multi_ok'] else 'FAIL'}"
         if open(f"{N}/func-{name}.txt").read() else "quick (decode only)"))
EOF

# late DFlash2 block (added 21:40 UTC): run once right after the Intel speed test, see df_hook.sh
if [ "$NAME" = I_best ] && [ -f $N/D3b.applied ] && [ ! -f $N/df_hook.done ]; then bash $N/df_hook.sh; fi
