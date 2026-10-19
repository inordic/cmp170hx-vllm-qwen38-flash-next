#!/bin/bash
# Night 3 extra experiments on server C (qfn-asgard-v2 + A1/A2, --trust-request-chat-template, port 8008).
# All on BFCL multi_turn_base x2 repeats (the only BFCL subset with real conversation history), paired with
# tpl-v2final / tpl-v2-nopreserve from verify.sh:
#   x-swiftorig         model's own template, reasoning sent back      (history handling: ours vs the model's)
#   x-v2-noreason       qfn-asgard-v2, client does NOT send reasoning back
#   x-swiftorig-noreason model's own template, client does NOT send reasoning back (empty <think> blocks)
# Deadline: no new experiment starts after DEADLINE (UTC HH:MM), so the production deploy is done before morning.
set -uo pipefail
N=/root/tune/night3; NL=$N/NIGHT-LOG.md; Q=/root/qeval; DEADLINE=${DEADLINE:-03:55}
note() { echo "- $(date -u +%T) UTC [exp] $*" >> $NL; }
late() { [ "$(date -u +%H%M)" \> "${DEADLINE/:/}" ]; }
K=$(cat /root/llm-api-key)

python3 - <<'EOF'
import json
t = open("/root/tpl-eval/swift-original.jinja").read()
json.dump({"chat_template": t}, open("/root/tune/night3/body-swiftorig.json", "w"))
EOF
# probe: the server must accept a request-level template
code=$(curl -s -o $N/exp-probe.json -w "%{http_code}" -H "Authorization: Bearer $K" -H "Content-Type: application/json" \
  -d "$(python3 -c 'import json; b=json.load(open("/root/tune/night3/body-swiftorig.json")); b.update(model="qwen3.8-flash-next", messages=[{"role":"user","content":"Say OK."}], max_tokens=200); print(json.dumps(b))')" \
  http://127.0.0.1:8008/v1/chat/completions)
note "request-level template probe: HTTP $code"
SWIFT_OK=0; [ "$code" = 200 ] && SWIFT_OK=1

run() {  # TAG EXTRA_BODY EXTRA_ARGS
  if late; then note "skip $1: past deadline $DEADLINE"; return; fi
  (cd $Q && ENDPOINT=http://127.0.0.1:8008 KEY_FILE=/root/llm-api-key MODEL=qwen3.8-flash-next TAG=$1 PHASES="bfcl" \
     BFCL_SPEC="multi_turn_base:100" BFCL_REPEATS=2 B_BFCL=30 EXTRA_BODY="$2" EXTRA_ARGS="$3" REASONING_EFFORT=medium \
     CONCURRENCY=16 ./run_tonight.sh >> $N/suite-$1.log 2>&1)
  note "$1: $(grep -h summary $Q/results/$1/medium/suite.log | cut -c1-160 | tr '\n' ';')"
}
# priority order: the empty-think question (no published measurement anywhere), then the model's own history
# handling, then the effort default (xhigh, capped at 40 min, BFCL x1 + EvalPlus, paired with tpl-v2final)
run tpl-x-v2-noreason '{"chat_template_kwargs":{"tool_prompt":"qwen"}}' "--history-reasoning none"   # same tool text as the model's own template
[ $SWIFT_OK = 1 ] && run tpl-x-swiftorig-noreason "@$N/body-swiftorig.json" "--history-reasoning none"
[ $SWIFT_OK = 1 ] && run tpl-x-swiftorig "@$N/body-swiftorig.json" ""
if late; then note "skip tpl-x-xhigh: past deadline $DEADLINE"; else
  (cd $Q && ENDPOINT=http://127.0.0.1:8008 KEY_FILE=/root/llm-api-key MODEL=qwen3.8-flash-next TAG=tpl-x-xhigh \
     PHASES="bfcl evalplus" BFCL_REPEATS=1 SUITE_MIN=40 B_BFCL=25 B_EVALPLUS=12 REASONING_EFFORT=xhigh \
     CONCURRENCY=16 ./run_tonight.sh >> $N/suite-tpl-x-xhigh.log 2>&1)
  note "tpl-x-xhigh: $(grep -h summary $Q/results/tpl-x-xhigh/xhigh/suite.log | cut -c1-160 | tr '\n' ';')"
fi
note "experiments DONE"
