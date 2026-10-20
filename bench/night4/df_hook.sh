#!/bin/bash
# Night 4 late DFlash2 block (added 2026-10-19 21:40 UTC). The P3c phase hit the Qwen4Exp spec-method guard (fixed
# by patch D3b after P3c had passed); run.sh cannot be edited while it runs, so speed.sh calls this once, right after
# the Intel speed test (P11), with the GPUs free of anything else. Report only.
# Swift + DFlash2 3 drafts: full speed set, greedy identity vs T0, dead-drafter check, tap/embedding log lines;
# retry (deploy4.sh, after patch D3c): Swift 3 drafts as above + Swift 4 drafts (decode speed); Intel already ran.
set -uo pipefail
N=/root/tune/night4; P=$N/profiles; NL=$N/NIGHT-LOG.md; BASE=$N/qfn-prod.env.baseline; M=/opt/vllm-qfn/models
LOG=/opt/vllm-qfn/logs/vllm.log; VPY=/opt/vllm-qfn-n4/.venv/bin/python; DF=$M/dflash2-flashnext
DEADLINE=0415
touch $N/df_hook.done
note() { echo "- $(date -u +%T) UTC $*" >> $NL; }
late() { [ "$(date -u +%H%M)" \> "$DEADLINE" ] && [ "$(date -u +%H%M)" \< "1200" ]; }
SPEC="spec=method:dflash spec=model:$DF spec=ngram_lookup:false spec=draft_tensor_parallel_size:1 spec=attention_backend:FLASH_ATTN mtp=3"
mk() { python3 $N/mkprof.py $BASE $P/$1.env venv=n4 port=8008 label=$1 "${@:2}" > /dev/null || note "[$1] profile build FAILED"; }
gpus_free() {
  for _ in $(seq 1 60); do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s += ($1 >= 1000)} END {exit s}' && return 0
    sleep 2
  done; return 1
}
stopall() { systemctl stop qfn-vllm 2>/dev/null; systemctl stop vllm-qfn 2>/dev/null; gpus_free; }
up() {
  if /opt/vllm-qfn/qfn-window.sh up $P/$1.env > $N/window-$1.log 2>&1; then
    note "[$1] up ($(grep -a -m1 -o 'ready after [0-9]* s' $N/window-$1.log)); $(grep -a -m1 -o 'GPU KV cache size: [0-9,]* tokens' $N/window-$1.log)"
    return 0
  fi
  note "[$1] server did NOT start: $(grep -a -iE 'error|exception|assert' $LOG | grep -v -i warn | tail -1 | cut -c1-260)"
  return 1
}
cmpres() {
  $VPY -c "
import json
def ld(n):
    try: return json.load(open('$N/res-' + n + '.json'))
    except Exception: return {}
a, b = ld('$1'), ld('$2')
f = lambda k: (f'{b[k]:.1f} ({b[k] / a[k] - 1:+.1%})' if a.get(k) and b.get(k) else 'n/a')
print(f'$2 vs $1: code {f(\"code\")}, edit {f(\"edit_off\")}, accept {b.get(\"accept\", 0):.2f} vs {a.get(\"accept\", 0):.2f}')"
}

late && { note "late DFlash2 block skipped (deadline)"; exit 0; }
note "late DFlash2 block (patch D3b lifted the Qwen4Exp spec-method guard)"
mk DF3 pmu=16 $SPEC
stopall
if up DF3; then
  note "[DF3] tap: $(grep -a -c 'Asgard patch D3: DFlash2 tap active' $LOG) rank lines; $(grep -a -o -m1 'target embed_tokens is [a-z ]*loaded' $LOG); $(grep -a -o -m1 'Eagle3 auxiliary layers from [a-z]*: ([0-9, ]*)' $LOG); draft graphs: $(grep -a -c 'Capturing model for DFlash2' $LOG)"
  bash $N/speed.sh DF3
  python3 -I $N/greedy_identity.py collect DF3 8008 >> $N/greedy.log 2>&1
  note "$(python3 -I $N/greedy_identity.py compare T0 DF3 2>&1 | tail -1)"
  python3 -I $N/T1-mtp-drafter-check.py --base http://127.0.0.1:8008 --model qwen3.8-flash-next --api-key "$(cat /root/llm-api-key)" \
    --phases 1,3,5,6,7 --temperature 0 > $N/T1-DF3.txt 2>&1
  note "[DF3] T1 rc=$?: $(tail -2 $N/T1-DF3.txt | tr '\n' ' ' | cut -c1-250)"
  note "DFlash2 vs MTP4 winner (report only): $(cmpres D4p16 DF3) | vs baseline MTP3+ngram: $(cmpres T0 DF3)"
fi
if ! late; then   # 4 drafts = the same depth as the MTP4 winner (5 is blocked by the QSA ring limit)
  mk DF4 pmu=16 ${SPEC/mtp=3/mtp=4}
  stopall
  up DF4 && QUICK=1 bash $N/speed.sh DF4 && note "DFlash2 4 drafts vs MTP4 winner (report only): $(cmpres D4p16 DF4)"
fi
stopall
note "late DFlash2 block done"
