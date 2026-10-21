#!/bin/bash
# Night 5 (2026-10-20/21): "maximum quality and speed". Reference R = today's production (int4 head + MTP4, Swift INT4
# attention, BF16 KV). Candidates add one piece each:
#   C1 = R + BF16 attention (Swift-A16) + INT8 KV (K1)     quality + capacity
#   C2 = C1 + probabilistic drafting                        speed at T=1
#   C3 = C2 + terse as template default                     shorter answers
# Every build gets the same full quality suite (BFCL x3, EvalPlus, long context) at T=1 on the same night.
# decide.py deploys the most complete candidate that passes quality, speed (task latency) and correctness gates;
# if none passes, production stays as it is. DEPLOY=0 (default since 2026-10-20): test and report only. Clocks locked at 1440 MHz for the A/B phases, reset before the deploy.
set -uo pipefail
N=${N5DIR:-/root/tune/night5}; P=$N/profiles; NL=$N/NIGHT-LOG.md; Q=/root/qeval; BASE=$N/qfn-prod.env.baseline
M=/opt/vllm-qfn/models; A16=$M/swift15-a16; VPY=/opt/vllm-qfn-n4/.venv/bin/python; LOG=/opt/vllm-qfn/logs/vllm.log
KV=kv=int8_per_token_head
DEADLINE=${DEADLINE:-0415}
DEPLOY=${DEPLOY:-0}   # 2026-10-20: the user made the night-4 setup the standard "for now" -> test and report only
note() { echo "- $(date -u +%T) UTC $*" >> $NL; }
late() { [ "$(date -u +%H%M)" \> "$DEADLINE" ] && [ "$(date -u +%H%M)" \< "1200" ]; }
mk() { python3 $N/mkprof.py $BASE $P/$1.env venv=n4 port=8008 label=$1 "${@:2}" > /dev/null || note "[$1] profile build FAILED"; }
mkall() {
  mk R
  mk C1 model=$A16 $KV
  mk C2 model=$A16 $KV spec=draft_sample_method:probabilistic
  mk C3 model=$A16 $KV spec=draft_sample_method:probabilistic tplkw=terse:true
}
gpus_free() {
  for _ in $(seq 1 60); do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s += ($1 >= 1000)} END {exit s}' && return 0
    sleep 2
  done; return 1
}
CUR=""
stopall() { systemctl stop qfn-vllm 2>/dev/null; systemctl stop vllm-qfn 2>/dev/null; gpus_free; CUR=""; }
lgc() {
  rm -f /run/n4-lgc.done; echo "$*" > /run/n4-lgc.req
  for _ in $(seq 1 18); do
    [ -f /run/n4-lgc.done ] && { note "[host] clocks: $(cut -c1-120 /run/n4-lgc.done)"; return 0; }
    sleep 5
  done
  rm -f /run/n4-lgc.req; note "[host] clock helper did not answer ($*)"; return 1
}
up() {
  if /opt/vllm-qfn/qfn-window.sh up $P/$1.env > $N/window-$1.log 2>&1; then
    note "[$1] up ($(grep -a -m1 -o 'ready after [0-9]* s' $N/window-$1.log)); $(grep -a -m1 -o 'GPU KV cache size: [0-9,]* tokens' $N/window-$1.log)"
    CUR=$1; return 0
  fi
  note "[$1] server did NOT start: $(grep -a -iE 'error|exception|assert' $LOG | grep -v -i warn | tail -1 | cut -c1-240)"
  return 1
}
speed() { bash $N/speed.sh "$@"; }
suite() {   # TAG -> full suite, BFCL x3 (budget raised so slower builds still finish the 3 repeats)
  (cd $Q && ENDPOINT=http://127.0.0.1:8008 KEY_FILE=/root/llm-api-key MODEL=qwen3.8-flash-next TAG=$1 \
     PHASES="bfcl evalplus longctx" EXTRA_BODY="" BFCL_REPEATS=3 B_BFCL=60 REASONING_EFFORT=medium CONCURRENCY=16 \
     ./run_tonight.sh >> $N/suite-$1.log 2>&1)
  note "[$1] suite: $(grep -h summary $Q/results/$1/medium/suite.log | sed 's/.*\[\(.*\)\] summary: {"n": \([0-9]*\), "acc": \([0-9.]*\).*runaway_rate": \([0-9.]*\).*/\1 n=\2 acc=\3 run=\4/' | tr '\n' ';')"
}
tx() {
  local t=$1; shift; local rc
  python3 -I $N/$t.py --base http://127.0.0.1:8008 --model qwen3.8-flash-next --api-key "$(cat /root/llm-api-key)" "$@" \
    > $N/$t-$CUR.txt 2>&1; rc=$?
  note "[$CUR] ${t%%-*} rc=$rc ($( [ $rc = 0 ] && echo PASS || echo FAIL )): $(tail -2 $N/$t-$CUR.txt | tr '\n' ' ' | cut -c1-260)"
  return $rc
}
chk() {   # TAG MODE PORT [ARGS] -> 0 when "ALL PASS"
  local tag=$1; shift
  python3 -I $N/k1_cow_check.py "$@" --log $LOG --out $N/$tag.json > $N/$tag.txt 2>&1
  note "[$tag] $(grep -a -E '^K1 ' $N/$tag.txt | tail -1) | $(grep -a -E '^(follow-up hits|\[pressure\] peak)' $N/$tag.txt | tr '\n' ' ' | cut -c1-300)"
  grep -a -q -E '^K1 .*ALL PASS' $N/$tag.txt
}
if [ "${DRY:-0}" = 1 ]; then   # control-flow dry run: stubs instead of servers and tests
  export N5DIR=$N QRES=$N/qres
  up() { note "[$1] DRY up"; CUR=$1; }; stopall() { CUR=""; }; lgc() { note "[host] DRY clocks $*"; }
  speed() { python3 -c "import json,sys; json.dump({'name':sys.argv[1],'code':190,'edit_off':300,'func_ok':True,'multi_ok':True}, open(sys.argv[2],'w'))" $1 $N/res-$1.json; note "[$1] DRY speed"; }
  suite() { note "[$1] DRY suite"; }; tx() { note "[$CUR] DRY $1"; }; chk() { note "[$1] DRY chk"; }
  bash() { note "DRY bash $*"; }; systemctl() { note "DRY systemctl $*"; }; curl() { echo 200; }
fi
trap 'lgc reset' EXIT

cp /opt/vllm-qfn/qfn-prod.env $BASE
echo "# Night 5 (2026-10-20/21): maximum quality and speed - R vs C1 (A16+INT8 KV), C2 (+probabilistic), C3 (+terse)" > $NL
note "start; production profile saved as $BASE"
rm -f $N/k1.ok
mkall
stopall; lgc lock 1440

# --- P1 reference: today's production ----------------------------------------------------------------------------
if up R; then
  speed R
  # full performance matrix of the standard config (repo README): prefill 1K-128K, users 8/16, mixed agent+chat
  PF=1 PERF_SIZES=1024,4096,8192,16384,32768,65536,131072 speed Rmatrix
  for sc in u8-1024 u16-1024 mixed; do
    out=$(cd /root/tune && python3 -I conc_stagger.py n5-R-$sc 8008 $sc 150 2>&1 | grep "^|" | tail -1); note "[R] conc $sc $out"
  done
  suite n5-R
fi

# --- P2 C1: A16 + INT8 KV; K1 checks first (first batch after start, multiturn, fanout, FULL pool, multiturn again) ---
if ! late; then
  stopall
  if up C1; then
    k1=0
    chk C1-first firstbatch 8008 --users 12 --doc-tokens 8000 || k1=1
    chk C1-mt multiturn 8008 --convs 6 --turns 5 --cold || k1=1
    chk C1-fan fanout 8008 --users 16 || k1=1
    chk C1-pinned pressure 8008 --convs 4 --fill 1.5 --flood-len 72000 --flood-conc 20 || k1=1
    chk C1-mt2 multiturn 8008 --convs 4 --turns 4 --seed 2 || k1=1
    note "[C1] server log: K1 CoW steps $(grep -a -c 'Asgard patch K1: CoW step' $LOG), NOT copied $(grep -a -c 'NOT copied (VLLM_K1_DISABLE' $LOG), tracebacks $(grep -a -c Traceback $LOG); GPU mem $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
    if [ $k1 = 0 ]; then
      touch $N/k1.ok; note "[C1] all K1 checks passed (INT8 KV usable)"
    else
      note "[C1] K1 checks FAILED -> INT8 KV dropped, candidates rebuilt with BF16 KV"
      KV=""; mkall; stopall; up C1 || true
    fi
    [ "$CUR" = C1 ] && { speed C1; suite n5-C1; }
  fi
fi

# --- P3 C2: + probabilistic drafting (suites run at T=1, where it helps) -------------------------------------------
if ! late; then
  stopall
  if up C2; then
    speed C2
    tx T1-mtp-drafter-check --phases 1,3,5,6,7 --temperature 1 --top-p 0.95 --top-k 20
    tx T2-pad-probe --n 24 --bg 6
    suite n5-C2
  fi
fi

# --- P4 C3: + terse as the template default (the full "max" build) -------------------------------------------------
if ! late; then
  stopall
  if up C3; then
    speed C3
    tx T4-prefix-bleed-soak --cases 24 --conc 6
    for sc in u8-1024 u16-1024 mixed; do
      out=$(cd /root/tune && python3 -I conc_stagger.py n5-C3-$sc 8008 $sc 150 2>&1 | grep "^|" | tail -1); note "[C3] conc $sc $out"
    done
    suite n5-C3
  fi
fi

# --- P5 decide + deploy (gate + rollback in deploy5.sh) -------------------------------------------------------------
$VPY $N/decide.py R C1 C2 C3 > $N/decide.txt 2>&1
while read -r l; do note "decide: $l"; done < $N/decide.txt
PICK=$(awk '/^PICK/ {print $2}' $N/decide.txt)
stopall
lgc reset
if [ "$DEPLOY" != 1 ]; then
  note "test-only night: decide would pick ${PICK:-none}; production stays on the standard (night-4) profile"
  systemctl start vllm-qfn
  for _ in $(seq 1 240); do [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8080/health)" = 200 ] && break; sleep 5; done
elif [ -n "$PICK" ] && [ "$PICK" != none ]; then
  note "deploying $PICK"
  bash $N/deploy5.sh $PICK
else
  note "no candidate passed every gate -> production unchanged"
  systemctl start vllm-qfn
  for _ in $(seq 1 240); do [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8080/health)" = 200 ] && break; sleep 5; done
fi
note "production 8080: $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8080/health)"
note "DONE"
