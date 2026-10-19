#!/bin/bash
# Night 3b: verify template v2 + vLLM patches A1/A2 with the chosen server defaults. usage: verify.sh DEFAULTS_JSON
# (DEFAULTS_JSON = the --default-chat-template-kwargs value, e.g. {"reasoning_effort":"medium","payload_rules":true})
set -uo pipefail
N=/root/tune/night3; P=$N/profiles; NL=$N/NIGHT-LOG.md; Q=/root/qeval; D="$1"
note() { echo "- $(date -u +%T) UTC [verify] $*" >> $NL; }
cd /opt/vllm-qfn-exp/src
for p in A1-thinking-off-consistency A2-history-reasoning-fields; do
  grep -q "Asgard patch ${p%%-*}" vllm/entrypoints/openai/chat_completion/serving.py vllm/entrypoints/chat_utils.py && { note "patch $p already applied"; continue; }
  /usr/bin/python3 /root/tpl-eval/vllm-patches/$p.py /opt/vllm-qfn-exp/src >> $N/verify.log 2>&1 && note "patch $p applied" || { note "patch $p FAILED"; exit 1; }
done
/opt/vllm-qfn-exp/.venv/bin/python -m py_compile vllm/entrypoints/openai/chat_completion/serving.py vllm/entrypoints/chat_utils.py || { note "patched files do not compile"; exit 1; }
cp /root/tpl-eval/qfn-asgard-v2.jinja /opt/vllm-qfn/templates/qfn-asgard-v2.jinja
DE=$(printf "%s" "$D" | sed "s/\"/\\\\\\\\\"/g")
sed -e "s#--chat-template /opt/vllm-qfn/templates/qfn-asgard-v1.jinja#--chat-template /opt/vllm-qfn/templates/qfn-asgard-v2.jinja#" \
    -e "s#--default-chat-template-kwargs {[^ ]*}#--default-chat-template-kwargs $DE#" -e "s#--enable-expert-parallel #--enable-expert-parallel --trust-request-chat-template #" $P/T_merged.env > $P/T_v2.env
note "profile T_v2: $(grep -o "chat-template [^ ]*\|default-chat-template-kwargs [^ ]*" $P/T_v2.env | tr "\n" " ")"
/opt/vllm-qfn/qfn-window.sh up $P/T_v2.env > $N/window-C.log 2>&1 || { note "server C did not start"; exit 1; }
cd /root/tpl-eval && python3 -I feature_check.py 8008 $N/feature_check-v2.json > $N/feature-v2.txt 2>&1; note "feature check: $(tail -1 $N/feature-v2.txt); fails: $(grep FAIL $N/feature-v2.txt | cut -c1-90 | tr "\n" ";")"
cd /root/tune
for c in prod_check func_check; do out=$(python3 -I $c.py 8008 2>&1); note "$c: $(echo "$out" | tail -1)"; done
out=$(python3 -I multiturn_check.py 8008 3 20000 2>&1); note "multiturn: $(echo "$out" | tail -2 | tr "\n" " ")"
(cd $Q && ENDPOINT=http://127.0.0.1:8008 KEY_FILE=/root/llm-api-key MODEL=qwen3.8-flash-next TAG=tpl-v2final PHASES="bfcl evalplus longctx" BFCL_REPEATS=2 REASONING_EFFORT=medium CONCURRENCY=16 ./run_tonight.sh >> $N/suite-tpl-v2final.log 2>&1)
note "tpl-v2final: $(grep -h summary $Q/results/tpl-v2final/medium/suite.log | sed "s/.*\[\(.*\)\] summary: {\"n\": \([0-9]*\), \"acc\": \([0-9.]*\).*runaway_rate\": \([0-9.]*\).*/\1 n=\2 acc=\3 run=\4/" | tr "\n" ";")"
# experiment: preserve_thinking=false (model default: drop reasoning before the last user turn), paired with tpl-v2final multi_turn_base
(cd $Q && ENDPOINT=http://127.0.0.1:8008 KEY_FILE=/root/llm-api-key MODEL=qwen3.8-flash-next TAG=tpl-v2-nopreserve PHASES="bfcl" BFCL_SPEC="multi_turn_base:100" BFCL_REPEATS=2 EXTRA_BODY='{"chat_template_kwargs":{"preserve_thinking":false}}' REASONING_EFFORT=medium CONCURRENCY=16 ./run_tonight.sh >> $N/suite-tpl-v2-nopreserve.log 2>&1)
note "tpl-v2-nopreserve: $(grep -h summary $Q/results/tpl-v2-nopreserve/medium/suite.log | cut -c1-200 | tr "\n" ";")"
note "DONE"
