#!/bin/bash
# Night 3b orchestrator: verify (patches, template v2, checks, suite, preserve_thinking experiment) -> extra
# experiments (deadline-guarded) -> gated production deploy. usage: night3b.sh DEFAULTS_JSON
set -uo pipefail
N=/root/tune/night3; D="$1"
echo "- $(date -u +%T) UTC [night3b] start, server defaults $D" >> $N/NIGHT-LOG.md
bash $N/verify.sh "$D" || echo "- $(date -u +%T) UTC [night3b] verify.sh exited $?" >> $N/NIGHT-LOG.md
if grep -q "\[verify\] DONE" $N/NIGHT-LOG.md; then
  bash $N/experiments.sh
  bash $N/deploy.sh "$D"
else
  echo "- $(date -u +%T) UTC [night3b] verify did not finish -> restoring baseline production, no deploy" >> $N/NIGHT-LOG.md
  /opt/vllm-qfn/qfn-window.sh prod >> $N/window-prod.log 2>&1
fi
echo "- $(date -u +%T) UTC [night3b] END; 8080 $(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:8080/health)" >> $N/NIGHT-LOG.md
