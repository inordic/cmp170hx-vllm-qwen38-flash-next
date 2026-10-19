#!/usr/bin/env bash
# One model, one reasoning effort: the full overnight quality suite (~2h20m, hard cap SUITE_MIN).
#
# Usage (on the server, after the model under test is up on ENDPOINT):
#   ENDPOINT=http://127.0.0.1:8080 KEY_FILE=/root/llm-api-key MODEL=qwen3.8-flash-next \
#   TAG=swift15 REASONING_EFFORT=medium CONCURRENCY=16 ./run_tonight.sh
#
# Results: $OUT/$TAG/$REASONING_EFFORT/{evalplus,bfcl,lcb,longctx}.jsonl (+ .summary.json, .meta.json)
# Resumable: re-running the same command skips items that are already in the .jsonl files.
set -uo pipefail
cd "$(dirname "$0")"

: "${ENDPOINT:=http://127.0.0.1:8080}"
: "${MODEL:=qwen3.8-flash-next}"
: "${TAG:?set TAG, e.g. swift15 / intel / cyankiwi}"
: "${REASONING_EFFORT:=medium}"
: "${CONCURRENCY:=16}"
: "${LONGCTX_CONCURRENCY:=6}"   # items of one haystack share the prefix KV, so 6 x 110K fits
: "${OUT:=$PWD/results}"
: "${PY:=python3}"
: "${SUITE_MIN:=150}"          # hard cap for the whole suite (minutes)
: "${TEMPERATURE:=1.0}"
: "${TOP_P:=0.95}"
: "${TOP_K:=20}"
# per-phase budgets in minutes (no new items are started after the budget; in-flight ones finish)
: "${B_EVALPLUS:=18}"
: "${B_BFCL:=38}"
: "${B_LCB:=50}"
: "${B_LONGCTX:=20}"
: "${BFCL_SPEC:=simple_python:100,multiple:100,parallel:100,parallel_multiple:100,irrelevance:100,live_simple:100,live_multiple:150,live_parallel:16,live_parallel_multiple:24,live_irrelevance:150,multi_turn_base:100}"
: "${LCB_RELEASES:=v6}"
: "${PHASES:=evalplus bfcl lcb longctx}"

if [[ -z "${KEY:-}" && -n "${KEY_FILE:-}" ]]; then KEY="$(cat "$KEY_FILE")"; fi
export KEY
LOGDIR="$OUT/$TAG/$REASONING_EFFORT"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/suite.log"
START=$(date +%s)
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

COMMON=(--endpoint "$ENDPOINT" --model "$MODEL" --tag "$TAG" --effort "$REASONING_EFFORT"
        --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" --out "$OUT")
[[ -n "${EXTRA_BODY:-}" ]] && COMMON+=(--extra-body "$EXTRA_BODY")   # e.g. chat_template_kwargs for template A/B
[[ -n "${EXTRA_ARGS:-}" ]] && COMMON+=($EXTRA_ARGS)   # e.g. --history-reasoning none

remaining() { echo $(( SUITE_MIN - ( $(date +%s) - START ) / 60 )); }
budget() {  # min(phase budget, remaining - 3)
  local b=$1 r; r=$(( $(remaining) - 3 )); (( r < b )) && b=$r; echo "$b"
}

log "=== suite start TAG=$TAG MODEL=$MODEL EFFORT=$REASONING_EFFORT C=$CONCURRENCY endpoint=$ENDPOINT"
ST_OUT="$($PY run_eval.py selftest 2>&1)"; echo "$ST_OUT" >> "$LOG"; echo "$ST_OUT" | tail -3
if ! grep -q "SELFTEST PASSED" <<<"$ST_OUT"; then log "sandbox selftest failed - aborting"; exit 2; fi
$PY run_eval.py probe "${COMMON[@]}" 2>&1 | tee -a "$LOG"

for phase in $PHASES; do
  case $phase in evalplus) pb=$B_EVALPLUS;; bfcl) pb=$B_BFCL;; lcb) pb=$B_LCB;; longctx) pb=$B_LONGCTX;; *) pb=30;; esac
  b=$(budget "$pb")
  if (( b < 5 )); then log "skip $phase: only $(remaining) min left"; continue; fi
  log "--- $phase (budget ${b} min, $(remaining) min left in suite)"
  case $phase in
    evalplus) $PY run_eval.py run --bench evalplus "${COMMON[@]}" --concurrency "$CONCURRENCY" --budget-min "$b" ;;
    bfcl)     $PY run_eval.py run --bench bfcl --bfcl-spec "$BFCL_SPEC" --repeats "${BFCL_REPEATS:-1}" "${COMMON[@]}" --concurrency "$CONCURRENCY" --budget-min "$b" ;;
    lcb)      $PY run_eval.py run --bench lcb --releases "$LCB_RELEASES" "${COMMON[@]}" --concurrency "$CONCURRENCY" --budget-min "$b" ;;
    longctx)  $PY run_eval.py run --bench longctx "${COMMON[@]}" --concurrency "$LONGCTX_CONCURRENCY" --budget-min "$b" ;;
    *) log "unknown phase $phase" ;;
  esac 2>&1 | grep -v '^\[sandbox\]' | tee -a "$LOG" | grep -E "summary|budget|items x|ERROR|Traceback" || true
done
log "=== suite done in $(( ( $(date +%s) - START ) / 60 )) min; results in $LOGDIR"
