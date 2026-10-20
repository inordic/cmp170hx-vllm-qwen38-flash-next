#!/bin/bash
# Night 4 (2026-10-19/20), plan v2: every key finding of "QFN speed and quality levers" plus the Q6-class quality
# builds. Speed phases run first (they choose what is deployed); quality builds, INT8 KV and terse are reported only.
# Clocks are locked (host helper) during the A/B phases so runs compare fairly, and reset before the deploy.
# Deploy = the greedy-draft speed winner (int4 head / MTP depth / batch / split) with the gate, else production back.
set -uo pipefail
N=${N4DIR:-/root/tune/night4}; P=$N/profiles; NL=$N/NIGHT-LOG.md; Q=/root/qeval; BASE=$N/qfn-prod.env.baseline
M=/opt/vllm-qfn/models
SH=$M/draft-heads/swift15-awq-lm_head-w4a16.safetensors
IH=$M/draft-heads/intel-autoround-lm_head-w4a16.safetensors
INTEL=$M/intel-autoround; A16=$M/swift15-a16; Q6X=$M/swift15-q6x
VPY=/opt/vllm-qfn-n4/.venv/bin/python
# switches of the night-4 patches in the n4 tree (mkprof args); empty = that test is skipped
MTP4_SWIFT=${MTP4_SWIFT:-$M/swift15-awq-mtp4}       # INT4 MTP experts (drafter 4.86 -> 1.39 GiB on PP1, the KV-limiting stage)
MTP4_INTEL=${MTP4_INTEL:-$M/intel-autoround-mtp4}   # same for Intel (hampsonw delta, config keys fixed for our mtp.py)
DF=$M/dflash2-flashnext                               # incoai DFlash2 drafter (block diffusion, 5 layers, 1.2 GB)
DF_SPEC=${DF_SPEC:-"spec=method:dflash spec=model:$DF spec=ngram_lookup:false spec=draft_tensor_parallel_size:1 spec=attention_backend:FLASH_ATTN"}
DEADLINE=${DEADLINE:-0415}     # UTC HHMM: no new phase starts after this; go straight to deploy/restore
CLOCK=${CLOCK:-1440}           # MHz, both cards, during the A/B phases
note() { echo "- $(date -u +%T) UTC $*" >> $NL; }
late() { [ "$(date -u +%H%M)" \> "$DEADLINE" ] && [ "$(date -u +%H%M)" \< "1200" ]; }
V=venv=n4; N4OK=1   # n4 tree (night-4 patches); falls back to the production (exp) tree if it does not start
mk() { python3 $N/mkprof.py $BASE $P/$1.env $V port=8008 label=$1 "${@:2}" > /dev/null || note "[$1] profile build FAILED"; }
argsof() {   # NAME [keys regex] -> the mkprof args a profile was built with (minus venv/port/label)
  sed -n 's/^# NIGHT 4 test profile .* by mkprof.py //p' $P/$1.env | sed 's/)$//' | tr ' ' '\n' \
    | grep -vE '^(venv|port|label)=' | grep -E "^(${2:-[a-z]+})=" | tr '\n' ' '
}
gpus_free() {
  for _ in $(seq 1 60); do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s += ($1 >= 1000)} END {exit s}' && return 0
    sleep 2
  done; return 1
}
CUR=""
stopall() { systemctl stop qfn-vllm 2>/dev/null; systemctl stop vllm-qfn 2>/dev/null; gpus_free; CUR=""; }
lgc() {   # "lock MHZ" | "reset" through the host helper (timer n4-dropcache on the Proxmox host, every 15 s)
  rm -f /run/n4-lgc.done; echo "$*" > /run/n4-lgc.req
  for _ in $(seq 1 18); do
    [ -f /run/n4-lgc.done ] && { note "[host] clocks: $(cut -c1-160 /run/n4-lgc.done)"; return 0; }
    sleep 5
  done
  rm -f /run/n4-lgc.req; note "[host] clock helper did not answer ($*)"; return 1
}
up() {   # NAME -> 0 when healthy on 8008; logs start time and KV pool
  if /opt/vllm-qfn/qfn-window.sh up $P/$1.env > $N/window-$1.log 2>&1; then
    note "[$1] up ($(grep -a -m1 -o 'ready after [0-9]* s' $N/window-$1.log)); $(grep -a -m1 -o 'GPU KV cache size: [0-9,]* tokens' $N/window-$1.log)"
    CUR=$1; return 0
  fi
  note "[$1] server did NOT start: $(grep -a -iE 'error|exception|assert' /opt/vllm-qfn/logs/vllm.log | tail -1 | cut -c1-240)"
  return 1
}
ensure() { [ "$CUR" = "$1" ] && return 0; stopall; up $1; }   # reuse the running server if it is NAME
speed() { bash $N/speed.sh "$@"; }
cmpres() {   # A B -> code / edit-off decode and acceptance of B relative to A (from res-*.json)
  $VPY -c "
import json, sys
def ld(n):
    try: return json.load(open('$N/res-' + n + '.json'))
    except Exception: return {}
a, b = ld('$1'), ld('$2')
f = lambda k: (f'{b[k]:.1f} ({b[k] / a[k] - 1:+.1%})' if a.get(k) and b.get(k) else 'n/a')
print(f'$2 vs $1: code {f(\"code\")}, edit {f(\"edit_off\")}, accept {b.get(\"accept\", 0):.2f} vs {a.get(\"accept\", 0):.2f}')"
}
suite() {   # TAG PHASES EXTRA_BODY [REPEATS]
  (cd $Q && ENDPOINT=http://127.0.0.1:8008 KEY_FILE=/root/llm-api-key MODEL=qwen3.8-flash-next TAG=$1 PHASES="$2" \
     EXTRA_BODY="$3" BFCL_REPEATS=${4:-2} REASONING_EFFORT=medium CONCURRENCY=16 ./run_tonight.sh >> $N/suite-$1.log 2>&1)
  note "[$1] suite: $(grep -h summary $Q/results/$1/medium/suite.log | sed 's/.*\[\(.*\)\] summary: {"n": \([0-9]*\), "acc": \([0-9.]*\).*runaway_rate": \([0-9.]*\).*/\1 n=\2 acc=\3 run=\4/' | tr '\n' ';')"
}
greedy() { python3 -I $N/greedy_identity.py collect $1 8008 >> $N/greedy.log 2>&1; }
gcmp() { note "$(python3 -I $N/greedy_identity.py compare $1 $2 2>&1 | tail -1)"; }
dist() { python3 -I $N/dist_test.py collect $1 8008 ${2:-1500} >> $N/dist.log 2>&1; }
LOG=/opt/vllm-qfn/logs/vllm.log
chk() {   # TAG MODE PORT [ARGS]
  local tag=$1; shift
  python3 -I $N/k1_cow_check.py "$@" --log $LOG --out $N/$tag.json > $N/$tag.txt 2>&1
  note "[$tag] $(grep -a -E '^K1 ' $N/$tag.txt | tail -1) | $(grep -a -E '^(follow-up hits|cold controls|server log|\[pressure\] peak)' $N/$tag.txt | tr '\n' ' ' | cut -c1-500)"
}
k1lines() { note "[$1] server log: K1 registrations $(grep -a -c 'per-block KV side tensors' $LOG), CoW steps $(grep -a -c 'Asgard patch K1: CoW step' $LOG), NOT copied $(grep -a -c 'NOT copied (VLLM_K1_DISABLE' $LOG), partial copies $(grep -a -c 'differ between dst and src' $LOG), tracebacks $(grep -a -c Traceback $LOG)"; }
tx() {   # T1..T4 (agent A tools) against 8008: tx T2-pad-probe [args]
  local t=$1; shift; local rc
  python3 -I $N/$t.py --base http://127.0.0.1:8008 --model qwen3.8-flash-next --api-key "$(cat /root/llm-api-key)" "$@" \
    > $N/$t-$CUR.txt 2>&1; rc=$?
  note "[$CUR] ${t%%-*} rc=$rc ($( [ $rc = 0 ] && echo PASS || echo FAIL/ERR )): $(tail -2 $N/$t-$CUR.txt | tr '\n' ' ' | cut -c1-300)"
}
canary() { note "$(python3 -I $N/canaries.py $1 8008 2>&1 | tail -6 | tr '\n' ';')"; }
pickpf() {   # REF NAME... -> best prefill variant (geomean 32K/128K prefill), needs >= +3% and code decode >= 98% of REF
  $VPY - "$@" <<'EOF'
import json, math, sys
import os; N = os.environ.get("N4DIR", "/root/tune/night4"); ref, names = sys.argv[1], sys.argv[2:]
def ld(n):
    try: return json.load(open(f"{N}/res-{n}.json"))
    except Exception: return None
def pf(r): return math.sqrt(r["pp"].get("32768", 0) * r["pp"].get("131072", 0)) if r and r.get("pp") else 0
R = ld(ref); best, bs = ref, pf(R)
for n in names:
    r = ld(n)
    if not r: print(f"skip {n}"); continue
    s = pf(r); c = r.get("code", 0) / max(1e-9, R.get("code", 1))
    print(f"{n}: prefill {s:.0f} ({s / max(1e-9, pf(R)) - 1:+.1%}), code decode {c:.0%}")
    if s > 1.03 * pf(R) and c >= 0.98 and s > bs: best, bs = n, s
print(f"PICK {best}")
EOF
}
if [ "${DRY:-0}" = 1 ]; then   # dry run of the control flow: stubs instead of servers and tests
  export N4DIR=$N
  up() { case " ${DRY_FAIL:-} " in *" $1 "*) note "[$1] DRY start failure"; return 1;; esac
         case " ${DRY_FAIL1:-} " in *" $1 "*) [ -f $N/.f-$1 ] || { touch $N/.f-$1; note "[$1] DRY start failure (once)"; return 1; };; esac
         note "[$1] DRY up ($(grep -o 'VLLM_VENV=[^ ]*' $P/$1.env))"; CUR=$1; }
  stopall() { CUR=""; }; lgc() { note "[host] DRY clocks $*"; }; dist() { :; }; greedy() { :; }; gcmp() { :; }; canary() { :; }
  tx() { note "[$CUR] DRY $*"; }; chk() { note "[$1] DRY chk $2"; }; k1lines() { :; }
  suite() { note "[$1] DRY suite $2"; }
  speed() { local n=${@: -1}; python3 -c "import json,random,sys; r=random.Random(sys.argv[1]); b=1.12 if sys.argv[1] in sys.argv[3].split() else 1.0; r.uniform=(lambda lo,hi,b=b: b); json.dump({'name':sys.argv[1],'pp':{'8192':7000*r.uniform(.95,1.1),'32768':7800*r.uniform(.95,1.1),'131072':8300*r.uniform(.95,1.1)},'code':169*r.uniform(.95,1.15),'edit_off':251*r.uniform(.95,1.15),'func_ok':True,'multi_ok':True}, open(sys.argv[2],'w'))" $n $N/res-$n.json "${DRY_BOOST:-}"; note "[$n] DRY speed $*"; }
  systemctl() { note "DRY systemctl $*"; }
  python3() { if [ "${1:-}" = "$N/select.py" ] || [ "${1:-}" = "$N/mkprof.py" ] || [ "${1:-}" = "-c" ]; then command python3 "$@"; else note "DRY python3 $*"; fi; }
  bash() { note "DRY bash $*"; }
  curl() { echo 200; }
fi
trap 'lgc reset' EXIT

echo "# Night 4 (2026-10-19/20) plan v2: drafting, bias test, batch/split, bounded prefill, A16, Q6x, Intel, INT8 KV, terse" > $NL
note "start; production profile saved as $BASE; clocks locked at $CLOCK MHz for the A/B phases"
mk T0
mk NOSPEC nospec=1
mk D3 head=$SH
mk D4 head=$SH mtp=4;  mk D4p16 head=$SH mtp=4 pmu=16
mk D5 head=$SH mtp=5;  mk D5p16 head=$SH mtp=5 pmu=16
stopall; lgc lock $CLOCK

# --- P0a offline GPU test (no server): probabilistic draft + rejection kernels unbiased; must flag the old shared key
[ "${DRY:-0}" = 1 ] && echo "DRY T3" > $N/T3.txt || CUDA_VISIBLE_DEVICES=0 timeout 900 $VPY $N/T3-spec-kernel-exactness.py > $N/T3.txt 2>&1
note "T3 kernel exactness rc=$?: $(grep -aiE 'shared|tree|c3|c1|PASS|FAIL|BIAS' $N/T3.txt | tail -6 | tr '\n' ';' | cut -c1-500)"

# --- P0 baseline (production settings on the n4 tree with the correctness fixes) + canaries + references ----------
if ! up T0; then
  N4OK=0; V=""
  note "n4 tree did not start -> production (exp) tree for the rest; n4-only tests (int4 head, draft noise, D2, K1) skipped"
  mk T0; mk NOSPEC nospec=1; stopall; up T0
fi
if [ "$CUR" = T0 ]; then
  speed T0; greedy T0; dist T0
  canary preempt
  tx T1-mtp-drafter-check --phases 1,3,5,6,7 --temperature 0
  [ $N4OK = 1 ] && tx T2-pad-probe --n 24 --bg 6
  tx T4-prefix-bleed-soak --cases 24 --conc 6
fi

# --- P1 no drafting: reference output distribution, and greedy outputs without any drafter ---------------------------
if ! late; then
  stopall
  if up NOSPEC; then dist NOSPEC; greedy NOSPEC; QUICK=1 speed NOSPEC; gcmp NOSPEC T0; fi
fi

# --- P2 int4 draft head (D1) and P3 MTP depth 4/5 (prefix-match-unit 16 if 32 does not start) ----------------------
CANDS=""
if ! late && [ $N4OK = 1 ]; then
  stopall; up D3 && { speed D3; greedy D3; gcmp T0 D3; CANDS="$CANDS D3"; }
  for d in 4 5; do
    late && break
    stopall
    if up D$d; then speed D$d; CANDS="$CANDS D$d"
    elif up D${d}p16; then speed D${d}p16; CANDS="$CANDS D${d}p16"
    fi
  done
fi
python3 $N/select.py T0 $CANDS > $N/select.txt 2>&1
BEST=$(awk '/^BEST/ {print $2}' $N/select.txt); GAIN=$(awk '/^BEST/ {print $4}' $N/select.txt)
BEST=${BEST:-T0}; GAIN=${GAIN:-0}
note "speed selection (greedy drafting): $(tr '\n' ';' < $N/select.txt)"

# --- P3b INT4 MTP experts on the winner: quality-neutral (drafter), +3.47 GiB KV on PP1; take it unless >1% slower ---
if ! late && [ -f $MTP4_SWIFT/config.json ]; then
  mk MB $(argsof $BEST 'head|mtp|pmu') model=$MTP4_SWIFT; stopall
  if up MB; then
    speed MB; greedy MB; gcmp T0 MB
    v=$($VPY $N/accept_mb.py $BEST MB)
    note "INT4 MTP experts on $BEST: $v"
    case "$v" in ACCEPT*) BEST=MB; GAIN=$(python3 $N/select.py T0 MB 2>/dev/null | awk '/^BEST/ {print $4}'); GAIN=${GAIN:-0};; esac
  fi
fi
BA=$(argsof $BEST 'head|mtp|pmu|model')

# --- P3c DFlash2 drafter (report only; needs patch D3 = Flash-Next feature tap + PP relay, marker $N/D3.applied) ----
# lossless check (greedy identity vs T0), speed + acceptance, dead-drafter check; 5 drafts; Intel (the base-model quant)
if ! late && [ $N4OK = 1 ] && [ -f $DF/config.json ] && [ -f $N/D3.applied ]; then
  DB=$(argsof $BEST 'pmu|model')
  mk DF3 $DB $DF_SPEC mtp=3; stopall
  if up DF3; then
    speed DF3; greedy DF3; gcmp T0 DF3
    tx T1-mtp-drafter-check --phases 1,3,5,6,7 --temperature 0
    note "[DF3] tap: $(grep -a -c 'Asgard patch D3: DFlash2 tap active' $LOG) rank lines, embed: $(grep -a -o 'target embed_tokens is [a-z ]*loaded[a-z ]*' $LOG | head -1), aux: $(grep -a -o -m1 'Eagle3 auxiliary layers from [a-z]*: ([0-9, ]*)' $LOG)"
    if ! late; then mk DF5 $DB $DF_SPEC mtp=5; stopall; up DF5 && QUICK=1 speed DF5; fi
  fi
  if ! late; then
    DI=$INTEL; [ -f $MTP4_INTEL/config.json ] && DI=$MTP4_INTEL
    mk DFI model=$DI $(argsof $BEST pmu) $DF_SPEC mtp=3; stopall; up DFI && QUICK=1 speed DFI
  fi
  note "DFlash2 vs $BEST (report only): $(cmpres $BEST DF3) | 5 drafts: $(cmpres $BEST DF5)"
elif [ ! -f $N/D3.applied ]; then
  note "P3c DFlash2 skipped: patch D3 not applied"
fi

# --- P4 probabilistic drafting (independent draft noise is in the tree, vllm#54282; C1+C2 fix the padded rows) -----
# speed at T=0.6, output-distribution test vs no drafting, dead-drafter check at T=1, fresh-prompt probe, quick BFCL;
# control PRpad = stock padding (VLLM_SPEC_PAD_NEW_REQS=1, T2 should fail) + the PP align-skew probe (C4, expect 0).
if ! late && [ $N4OK = 1 ]; then
  mk PRn $BA spec=draft_sample_method:probabilistic
  stopall
  if up PRn; then
    speed PRn; dist PRn
    note "bias test: $(python3 -I $N/dist_test.py compare NOSPEC T0 PRn 2>&1 | tr '\n' ';')"
    tx T1-mtp-drafter-check --phases 1,3,5,6,7 --temperature 1 --top-p 0.95 --top-k 20
    tx T2-pad-probe --n 24 --bg 6
    suite n4-PRn "bfcl" "" 1
  fi
  if ! late; then
    mk PRpad $BA spec=draft_sample_method:probabilistic env=VLLM_SPEC_PAD_NEW_REQS=1 env=VLLM_PP_SKEW_PROBE=1
    stopall
    if up PRpad; then
      tx T2-pad-probe --n 24 --bg 6
      tx T4-prefix-bleed-soak --cases 12 --conc 6
      note "[PRpad] $(grep -a 'Asgard C4 PP skew probe' /opt/vllm-qfn/logs/vllm.log | tail -2 | cut -c1-200 | tr '\n' ';')"
    fi
  fi
fi

# --- P5 batch size (align mode: prefill chunks are multiples of 1600, +96 for 24 decodes x 4) and PP stage split ----
if ! late; then
  mk SW0 $BA; stopall; up SW0 && PF=1 speed SW0
  SWS=""
  # with the INT4 drafter PP1 carries ~2.6 GiB more than PP0 -> 25/23 balances; 26/22 only without it
  SPL="S25 split=25,23"; case "$BA" in *mtp4*) ;; *) SPL="S26 split=26,22";; esac
  for v in "B3296 batch=3296" "B4896 batch=4896" "$SPL"; do
    late && break
    set -- $v; mk $1 $BA ${@:2}; stopall; up $1 && { PF=1 speed $1; SWS="$SWS $1"; }
  done
  pickpf SW0 $(echo $SWS | tr ' ' '\n' | grep '^B') > $N/pick-batch.txt 2>&1
  pickpf SW0 $(echo $SWS | tr ' ' '\n' | grep '^S') > $N/pick-split.txt 2>&1
  PB=$(awk '/^PICK/ {print $2}' $N/pick-batch.txt); PS=$(awk '/^PICK/ {print $2}' $N/pick-split.txt)
  note "batch pick: $(tr '\n' ';' < $N/pick-batch.txt)"; note "split pick: $(tr '\n' ';' < $N/pick-split.txt)"
fi
XA=""
[ -n "${PB:-}" ] && [ "$PB" != SW0 ] && XA="$XA $(argsof $PB batch)"
[ -n "${PS:-}" ] && [ "$PS" != SW0 ] && XA="$XA $(argsof $PS split)"

# --- P6 prefill beside decodes: stock long-prefill threshold vs bounded mixed prefill (D2, >= 1600) under load ------
if ! late; then
  mk BP_off $BA $XA
  mk BP_lpt $BA $XA "addflag=--long-prefill-token-threshold 3200"
  mk BP_d2 $BA $XA chunk=1600
  for b in BP_off BP_lpt BP_d2; do
    late && break
    [ $b = BP_d2 ] && [ $N4OK = 0 ] && continue
    stopall
    if up $b; then
      for sc in mixed u8-1024; do
        out=$(cd /root/tune && python3 -I conc_stagger.py n4-$b 8008 $sc 150 2>&1 | grep "^|" | tail -1)
        note "[$b] conc $sc $out"
      done
    fi
  done
fi

# --- P7 combined candidate (greedy drafting winner + batch + split): full speed set + the gate checks --------------
FINAL=$BEST
if [ -n "$XA" ] && ! late; then
  mk FINAL $BA $XA; stopall
  if up FINAL; then
    speed FINAL; tx T4-prefix-bleed-soak --cases 24 --conc 6
    python3 -c "import json,sys; r=json.load(open('$N/res-FINAL.json')); sys.exit(0 if r.get('func_ok') and r.get('multi_ok') else 1)" \
      && FINAL=FINAL || note "FINAL failed func/multiturn -> keeping $BEST"
  fi
fi
note "deploy candidate: $FINAL ($(argsof $FINAL))"

# --- P8 same-night quality reference on the deploy candidate (Swift, greedy drafting) -------------------------------
if ! late; then
  ensure $FINAL && suite n4-ref "bfcl evalplus longctx" ""
fi
FA=$(argsof $FINAL 'head|mtp|pmu|batch|split')          # for other checkpoints (model= set per phase)
FAM=$(argsof $FINAL 'head|mtp|pmu|batch|split|model')    # the deploy candidate itself

# --- P9 Swift-A16 (BF16 attention, INT4 experts) and P10 Swift-Q6x (A16 + INT8 experts in layers 0-10 and 43-47) ---
# PP1 carries lm_head + MTP (~6 GiB more than PP0), so Q6x puts 11 INT8 layers on PP0 and 5 on PP1: keep the 24/24
# split for it and give it GPU memory 0.93 (the 196K-context retry covers a short KV pool).
FA_NOSPLIT=$(echo $FA | tr ' ' '\n' | grep -v '^split=' | tr '\n' ' ')
for q in "A16 $A16 $FA" "Q6X $Q6X $FA_NOSPLIT gpumem=0.93"; do
  set -- $q
  late && break
  [ -f "$2/config.json" ] || { note "[$1] model $2 missing - skipped"; continue; }
  mk $1 model=$2 ${@:3}; stopall
  if ! up $1; then   # less KV headroom: retry with a 196K context limit
    mk ${1}m model=$2 ${@:3} maxlen=196608; stopall
    up ${1}m || continue
    set -- ${1}m $2
  fi
  speed $1; suite n4-$1 "bfcl evalplus longctx" ""
done

# --- P11 Intel AutoRound + its int4 head + INT4 MTP experts (if installed): speed only (quality known, drafter-neutral)
if ! late; then
  IM=$INTEL; [ -f $MTP4_INTEL/config.json ] && IM=$MTP4_INTEL
  IA=$(argsof $FINAL 'mtp|pmu|batch|split'); [ -n "$(argsof $FINAL head)" ] && IA="$IA head=$IH"
  mk I_best model=$IM $IA; stopall; up I_best && speed I_best
  [ -f $N/res-DFI.json ] && note "Intel DFlash2 vs MTP (report only): $(cmpres I_best DFI)"
fi

# --- P12 INT8 KV with the K1 scale copy (always on with INT8; VLLM_K1_DISABLE=1 = the old bug, as control) ---------
# control must FAIL on cache-hit turns (else the check does not see the bug); debug run must show 0 partial copies;
# production-like run: first batch after start, multiturn, pool churn, speed, KV pool, long-context quality.
if ! late && [ $N4OK = 1 ]; then
  KV8="$FAM kv=int8_per_token_head"
  mk K8off $KV8 env=VLLM_K1_DISABLE=1 env=VLLM_K1_DEBUG=1 env=VLLM_K1_LOG_EVERY=1
  mk K8dbg $KV8 env=VLLM_K1_DEBUG=1 env=VLLM_K1_LOG_EVERY=1
  mk K8 $KV8
  stopall
  if up K8off; then chk K8off-mt multiturn 8008 --convs 4 --turns 4 --cold; k1lines K8off; fi
  if ! late; then
    stopall
    if up K8dbg; then
      chk K8dbg-mt multiturn 8008 --convs 4 --turns 4 --cold; chk K8dbg-fan fanout 8008 --users 12; k1lines K8dbg
    fi
  fi
  if ! late; then
    stopall
    if up K8; then
      chk K8-first firstbatch 8008 --users 12 --doc-tokens 8000
      chk K8-mt multiturn 8008 --convs 6 --turns 5 --cold
      chk K8-churn pressure 8008 --convs 4 --fill 1.2 --flood-len 32000 --flood-conc 8
      speed K8; k1lines K8
      suite n4-K8 "longctx" ""
    fi
  fi
fi

# --- P13 terse on its own (Swift, deploy candidate), full quality suite ---------------------------------------------
if ! late; then
  ensure $FINAL && suite n4-terse "bfcl evalplus" '{"chat_template_kwargs":{"terse":true}}' 1
fi

# --- P14 clocks back to default, deploy the speed winner with the gate, else restore production --------------------
stopall
lgc reset
FG=$GAIN
[ "$FINAL" = FINAL ] && FG=$(python3 $N/select.py T0 FINAL 2>/dev/null | awk '/^BEST/ {print $4}')
if [ "$FINAL" != T0 ] && python3 -c "import sys; sys.exit(0 if float('${FG:-0}') >= 0.05 or '$FINAL' in ('FINAL', 'MB') else 1)"; then
  note "deploying $FINAL (code/edit gain ${FG:-?} vs T0)"
  bash $N/deploy4.sh $FINAL
else
  note "no speed winner >= 5% (best $FINAL, gain ${FG:-0}) -> production unchanged"
  systemctl start vllm-qfn
  for _ in $(seq 1 240); do [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8080/health)" = 200 ] && break; sleep 5; done
fi
note "production 8080: $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8080/health)"
note "DONE"
