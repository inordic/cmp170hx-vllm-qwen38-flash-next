#!/bin/bash
# Night 3: chat-template A/B on Swift 1.5 (FINAL config, port 8008). Variants of qfn-asgard-v1 via chat_template_kwargs,
# baseline = chromix v1 (production template). BFCL x2 repeats (2,080 calls) for every variant; EvalPlus + long context
# where the non-tool prompt differs (v0, v2-terse, chromix).
N=/root/tune/night3; P=$N/profiles; NL=$N/NIGHT-LOG.md; Q=/root/qeval
note() { echo "- $(date -u +%T) UTC $*" >> $NL; }
suite() {  # TAG PHASES EXTRA_BODY
  (cd $Q && ENDPOINT=http://127.0.0.1:8008 KEY_FILE=/root/llm-api-key MODEL=qwen3.8-flash-next TAG=$1 PHASES="$2" \
     EXTRA_BODY="$3" BFCL_REPEATS=2 REASONING_EFFORT=medium CONCURRENCY=16 ./run_tonight.sh >> $N/suite-$1.log 2>&1)
  note "$1 done: $(grep -h summary $Q/results/$1/medium/suite.log | sed "s/.*\[\(.*\)\] summary: {\"n\": \([0-9]*\), \"acc\": \([0-9.]*\).*runaway_rate\": \([0-9.]*\).*/\1 n=\2 acc=\3 run=\4/" | tr "\n" ";")"
}
echo "# Night 3 (2026-10-16/17): best chat template for QFN - log" > $NL
note "start; server A = qfn-asgard-v1"
/opt/vllm-qfn/qfn-window.sh up $P/T_merged.env > $N/window-A.log 2>&1 || { note "server A did not start"; exit 1; }
suite tpl-v0 "bfcl evalplus longctx" "{\"chat_template_kwargs\":{}}"
suite tpl-v1 "bfcl" "{\"chat_template_kwargs\":{\"payload_rules\":true,\"failure_notes\":true}}"
suite tpl-v2 "bfcl evalplus longctx" "{\"chat_template_kwargs\":{\"payload_rules\":true,\"failure_notes\":true,\"terse\":true}}"
suite tpl-v3 "bfcl" "{\"chat_template_kwargs\":{\"strict_tool_rules\":true}}"
suite tpl-v4 "bfcl" "{\"chat_template_kwargs\":{\"payload_rules\":true,\"failure_notes\":true,\"strict_tool_rules\":true}}"
note "server B = chromix v1 (production template)"
/opt/vllm-qfn/qfn-window.sh up $P/T_chromix.env > $N/window-B.log 2>&1 || { note "server B did not start"; }
suite tpl-chromix "bfcl evalplus longctx" ""
(cd $Q && python3 run_eval.py compare results/tpl-chromix/medium results/tpl-v0/medium results/tpl-v1/medium results/tpl-v2/medium results/tpl-v3/medium results/tpl-v4/medium --labels chromix,v0,v1,v2,v3,v4 --json-out results/compare-templates.json >> $N/compare.log 2>&1)
note "comparison written: $Q/results/compare-templates.md"
/opt/vllm-qfn/qfn-window.sh prod > $N/window-prod.log 2>&1; note "production restored (baseline): 8080 $(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:8080/health)"
note "DONE"
