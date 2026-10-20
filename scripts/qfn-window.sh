#!/usr/bin/env bash
# qfn-window.sh - swap the GPUs between production Strata and the vLLM test engine.
#
#   qfn-window.sh up   [profile]   stop Strata, start vLLM (systemd unit qfn-vllm), wait for /health on 8008
#   qfn-window.sh prod             stop any test vLLM, start the permanent service vllm-qfn (8080), wait for health
#   qfn-window.sh status
set -uo pipefail
PROFILE="${2:-/opt/vllm-qfn/qfn-pp2.env}"
LOG=/opt/vllm-qfn/logs/vllm-$(date +%Y%m%d-%H%M%S).log

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' '; }

wait_http() {   # url timeout_s [unit] [log]: also gives up early when the unit exits or the engine failed to start
  local t0=$SECONDS
  until [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$1")" = "200" ]; do
    (( SECONDS - t0 > $2 )) && return 1
    if [ -n "${3:-}" ] && ! systemctl is-active -q "$3"; then echo "  $3 exited after $((SECONDS - t0)) s"; return 1; fi
    if [ -n "${4:-}" ] && grep -a -q -E "Engine core initialization failed|EngineCore failed to start" "$4" 2>/dev/null; then
      echo "  engine failed to start after $((SECONDS - t0)) s"; return 1
    fi
    sleep 5
  done
  echo "  ready after $((SECONDS - t0)) s"
}

case "${1:-}" in
up)
  echo "=== stopping any running vLLM (test unit and permanent service)"
  systemctl stop qfn-vllm 2>/dev/null || true
  systemctl stop vllm-qfn 2>/dev/null || true
  for _ in $(seq 1 60); do
    used=$(gpu_used); [ "$(echo "$used" | awk '{print ($1 < 1000 && $2 < 1000)}')" = 1 ] && break; sleep 2
  done
  echo "  GPU memory used (MiB): $(gpu_used)"
  echo "=== starting vLLM: $PROFILE  (log $LOG)"
  ln -sfn "$LOG" /opt/vllm-qfn/logs/vllm.log
  systemctl reset-failed qfn-vllm 2>/dev/null || true
  systemd-run --unit=qfn-vllm --collect -p LimitMEMLOCK=infinity -p LimitNOFILE=65536 -p KillMode=control-group \
    -p TimeoutStopSec=60 --setenv=RUN_ENV="$PROFILE" bash -c "exec bash /opt/vllm-qfn/release/serve/serve.sh >> '$LOG' 2>&1"
  VPORT=$(sed -n "s/^PORT=\([0-9]*\).*/\1/p" "$PROFILE" | tail -1); VPORT=${VPORT:-8008}
  if wait_http "http://127.0.0.1:$VPORT/health" 1200 qfn-vllm "$LOG"; then
    grep -a -E "^=== |KV cache size|Maximum concurrency" "$LOG" | head -30
  else
    echo "=== vLLM did not become healthy; last log lines:"; tail -40 "$LOG"; exit 1
  fi
  ;;
prod)
  echo "=== stopping test vLLM, starting the permanent service vllm-qfn"
  systemctl stop qfn-vllm 2>/dev/null || true
  for _ in $(seq 1 60); do
    used=$(gpu_used); [ "$(echo "$used" | awk '{print ($1 < 1000 && $2 < 1000)}')" = 1 ] && break; sleep 2
  done
  systemctl start vllm-qfn
  wait_http http://127.0.0.1:8080/health 1200 || { echo "=== vllm-qfn not healthy"; tail -40 /var/log/vllm-qfn/vllm.log; exit 1; }
  ;;
down)
  echo "Strata was removed on 2026-10-13 (archive: /root/strata-archive.tar.gz); use: $0 prod"; exit 2
  ;;
status)
  echo "vllm-qfn (prod): $(systemctl is-active vllm-qfn)   qfn-vllm (test): $(systemctl is-active qfn-vllm 2>/dev/null)"
  echo "GPU memory used (MiB): $(gpu_used)"
  ;;
*) echo "usage: $0 up [profile] | prod | status"; exit 2 ;;
esac
