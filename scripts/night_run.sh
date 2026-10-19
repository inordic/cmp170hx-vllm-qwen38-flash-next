#!/usr/bin/env bash
# night_run.sh NAME PROFILE TEST...  - start PROFILE as the test unit (port from the profile), run the tests,
# write the full output to /root/tune/night/run-NAME.log and summary lines to NIGHT-LOG.md.
#   tests: func prod exact code multi quality disk conc:<scenario>:<seconds>   (e.g. conc:u8-1024:120, conc:mixed:150)
set -uo pipefail
NAME=$1; PROFILE=$2; shift 2
N=${NIGHT_DIR:-/root/tune/night}; LOG=$N/run-$NAME.log; NL=$N/NIGHT-LOG.md
PORT=$(sed -n "s/^PORT=\([0-9]*\).*/\1/p" "$PROFILE" | tail -1)
MODEL=$(sed -n "s/^MODEL=//p" "$PROFILE" | tail -1)
note() { echo "- $(date -u +%T) UTC [$NAME] $*" >> $NL; echo "### $*" >> $LOG; }

# safety: stop the night if the root filesystem went read-only or is nearly full
if mount | grep " / " | grep -q emergency_ro || [ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -lt 30 ]; then
  note "ABORT: root filesystem read-only or < 30 GB free"; exit 3
fi

echo "##### $NAME  profile=$PROFILE  $(date -u)" > $LOG
diff $N/qfn-prod.env.baseline "$PROFILE" >> $LOG
started=0
for unit in "" 16 8; do
  if [ -n "$unit" ]; then   # the KV block size is not divisible by the prefix-match unit: retry with a smaller one
    grep -a -q "Invalid prefix_match_unit" /opt/vllm-qfn/logs/vllm.log || break
    sed -i "s/--prefix-match-unit [0-9]*/--prefix-match-unit $unit/" "$PROFILE"
    note "retry with --prefix-match-unit $unit ($(grep -a -o "block sizes=\[[0-9, ]*\]" /opt/vllm-qfn/logs/vllm.log | head -1))"
  fi
  if /opt/vllm-qfn/qfn-window.sh up "$PROFILE" >> $LOG 2>&1; then started=1; break; fi
done
if [ $started = 0 ]; then
  note "START FAILED: $(grep -a -E "Error|error" /opt/vllm-qfn/logs/vllm.log | grep -v INFO | tail -2 | tr '\n' ' ' | cut -c1-300)"
  exit 2
fi
VLOG=$(readlink -f /opt/vllm-qfn/logs/vllm.log)
note "started: $(grep -a -o "GPU KV cache size: [0-9,]* tokens" $VLOG | tail -1), $(grep -a -o "attention block size to [0-9]*" $VLOG | tail -1), load $(grep -a -o "init engine.*took [0-9.]* s" $VLOG | tail -1 | grep -o "[0-9.]* s$")"

nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,utilization.gpu,temperature.gpu --format=csv,noheader,nounits -lms 1000 > $N/clocks-$NAME.csv &
SMI=$!
cd /root/tune
for t in "$@"; do
  case "$t" in
    func)    out=$(python3 -I func_check.py $PORT 2>&1); echo "$out" >> $LOG; note "func: $(echo "$out" | grep -c PASS)/8 pass, $(echo "$out" | tail -1)" ;;
    prod)    out=$(python3 -I prod_check.py $PORT 2>&1); echo "$out" >> $LOG; note "prod: $(echo "$out" | tail -1); $(echo "$out" | grep "hard question" | sed 's/.*effort=//' | tr -s ' ' | tr '\n' ';')" ;;
    exact)   out=$(/opt/vllm-qfn/.venv/bin/python -I perf_exact.py "$NAME" $PORT "$MODEL" 3 256 2>&1); echo "$out" >> $LOG
             note "exact (prompt: TTFT, prefill tok/s, decode): $(echo "$out" | grep -E '^\| \| ' | awk -F'|' '{gsub(/ /,"",$3); gsub(/ /,"",$4); split($5,a," "); gsub(/ /,"",$6); printf "%s: %s %s %s; ", $3, $4, a[1], $6}')" ;;
    long)    out=$(PERF_SIZES=65536,131072 /opt/vllm-qfn/.venv/bin/python -I perf_exact.py "$NAME-long" $PORT "$MODEL" 2 64 2>&1); echo "$out" >> $LOG
             note "long (prompt: TTFT, prefill tok/s): $(echo "$out" | grep -E '^\| \| ' | awk -F'|' '{gsub(/ /,"",$3); gsub(/ /,"",$4); split($5,a," "); printf "%s: %s %s; ", $3, $4, a[1]}')" ;;
    code)    out=$(python3 -I decode_code.py "$NAME" $PORT 768 2>&1); echo "$out" >> $LOG; note "code decode: $(echo "$out" | grep '^| ' | cut -d'|' -f3,4)" ;;
    multi)   out=$(python3 -I multiturn_check.py $PORT 2 20000 2>&1); echo "$out" >> $LOG
             note "multiturn: $(echo "$out" | tail -1), follow-up TTFT $(echo "$out" | grep 'turn 2' | grep -o 'TTFT *[0-9.]*s' | tr '\n' ' '), $(echo "$out" | grep 'prefix cache' | grep -o '([0-9]*%)')" ;;
    quality) out=$(python3 -I eval_quality.py "$NAME" $PORT 2>&1); echo "$out" >> $LOG; note "quality: $(echo "$out" | grep '^| ' | cut -d'|' -f3-5)" ;;
    offload) out=$(python3 -I offload_test.py "$NAME" $PORT 40 32768 2>&1); echo "$out" >> $LOG; note "offload: $(echo "$out" | grep '^| ' | cut -d'|' -f3-)" ;;
    edit)    out=$(python3 -I edit_decode.py "$NAME" $PORT 2>&1); echo "$out" >> $LOG; note "code-edit decode: $(echo "$out" | grep '^| ' | cut -d'|' -f3,4 | tr '\n' ';')" ;;
    method)  out=$(python3 -I method_compare.py "$NAME" $PORT 3 2>&1); echo "$out" >> $LOG; note "random-token method (prompt: latency, tok/s): $(echo "$out" | grep -E '^\| \| ' | awk -F'|' '{gsub(/ /,"",$3); gsub(/ /,"",$4); split($5,a," "); printf "%s: %s %s; ", $3, $4, a[1]}')" ;;
    soak:*)  IFS=: read -r _ mins <<< "$t"; out=$(python3 -I soak_mixed.py $PORT $mins 2>&1); echo "$out" >> $LOG; note "$out" ;;
    disk)    out=$(python3 -I ple_disk.py $PORT 2>&1); echo "$out" >> $LOG; note "disk: $(echo "$out" | grep '^disk' | tr '\n' ' ')" ;;
    conc:*)  IFS=: read -r _ scen dur <<< "$t"; out=$(python3 -I conc_stagger.py "$NAME" $PORT "$scen" "$dur" 2>&1); echo "$out" >> $LOG
             note "conc $(echo "$out" | grep '^| ' | cut -d'|' -f3-)" ;;
  esac
done
kill $SMI 2>/dev/null
acc=$(grep -a -o "Mean acceptance length: [0-9.]*" $VLOG | awk '{s+=$4; n++} END {if (n) printf "%.2f (%d samples)", s/n, n}')
note "MTP mean acceptance ${acc:-n/a}; GPU busy clocks: $(python3 - $N/clocks-$NAME.csv <<'PY'
import csv, statistics, sys
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) == 6]
out = []
for g in ("0", "1"):
    b = [r for r in rows if r[0].strip() == g and float(r[4]) >= 50]
    if b:
        out.append(f"GPU{g} SM {statistics.median(float(r[1]) for r in b):.0f} MHz, {statistics.median(float(r[3]) for r in b):.0f} W, max {max(float(r[5]) for r in b):.0f} C")
print("; ".join(out))
PY
)"
echo "RUN DONE" >> $LOG
