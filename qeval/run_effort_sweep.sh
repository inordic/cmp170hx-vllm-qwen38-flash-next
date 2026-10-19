#!/usr/bin/env bash
# Quality vs reasoning_effort on ONE model (run after run_tonight.sh did the medium suite).
# A reduced suite per effort (~30-40 min each); its items are a subset of the main suite's items,
# so `run_eval.py compare results/$TAG/low results/$TAG/medium results/$TAG/high` pairs them.
#
#   ENDPOINT=... KEY_FILE=/root/llm-api-key MODEL=qwen3.8-flash-next TAG=swift15 \
#   EFFORTS="low high" CONCURRENCY=16 ./run_effort_sweep.sh
# Optional extras:  C1=1   -> single-stream latency probe (20 HumanEval+ items per effort, tag $TAG-c1)
#                   T06=1  -> temperature 0.6 ablation on HumanEval+ at medium (tag $TAG-t06)
set -uo pipefail
cd "$(dirname "$0")"
: "${ENDPOINT:=http://127.0.0.1:8080}"; : "${MODEL:=qwen3.8-flash-next}"; : "${TAG:?set TAG}"
: "${EFFORTS:=low high}"; : "${CONCURRENCY:=16}"; : "${OUT:=$PWD/results}"; : "${PY:=python3}"
: "${C1:=0}"; : "${T06:=0}"
: "${MINI_BFCL:=simple_python:50,multiple:50,parallel_multiple:50,irrelevance:50,live_multiple:50,live_irrelevance:50,multi_turn_base:50}"
if [[ -z "${KEY:-}" && -n "${KEY_FILE:-}" ]]; then KEY="$(cat "$KEY_FILE")"; fi
export KEY
mkdir -p "$OUT/$TAG"
LOG="$OUT/$TAG/effort_sweep.log"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
filt() { grep -v '^\[sandbox\]' | tee -a "$LOG" | grep -E "summary|budget|items x|Traceback" || true; }
for E in $EFFORTS; do
  C=(--endpoint "$ENDPOINT" --model "$MODEL" --tag "$TAG" --effort "$E" --out "$OUT")
  log "=== effort $E"
  $PY run_eval.py run --bench evalplus --subsets humaneval "${C[@]}" --concurrency "$CONCURRENCY" --budget-min 10 2>&1 | filt
  $PY run_eval.py run --bench bfcl --bfcl-spec "$MINI_BFCL" "${C[@]}" --concurrency "$CONCURRENCY" --budget-min 10 2>&1 | filt
  $PY run_eval.py run --bench lcb --limit 80 --max-tokens 32768 "${C[@]}" --concurrency "$CONCURRENCY" --budget-min 15 2>&1 | filt
  $PY run_eval.py run --bench longctx --per-bin 2 "${C[@]}" --concurrency 6 --budget-min 8 2>&1 | filt
done
if [[ "$C1" == 1 ]]; then
  for E in low medium high; do
    log "=== single-stream latency probe, effort $E"
    $PY run_eval.py run --bench evalplus --subsets humaneval --limit 20 --endpoint "$ENDPOINT" --model "$MODEL" \
        --tag "$TAG-c1" --effort "$E" --out "$OUT" --concurrency 1 2>&1 | filt
  done
fi
if [[ "$T06" == 1 ]]; then
  log "=== temperature 0.6 ablation (HumanEval+, medium)"
  $PY run_eval.py run --bench evalplus --subsets humaneval --endpoint "$ENDPOINT" --model "$MODEL" --tag "$TAG-t06" \
      --effort medium --temperature 0.6 --out "$OUT" --concurrency "$CONCURRENCY" 2>&1 | filt
fi
log "=== sweep done"
