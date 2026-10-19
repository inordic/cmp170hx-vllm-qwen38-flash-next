#!/bin/bash
# Night 3c: put chat template qfn-asgard-v2 (+ vLLM patches A1/A2, applied by verify.sh) into production on 8080
# with the chosen server defaults. Gate: prod/func/multiturn checks on 8080 must all pass, otherwise the baseline
# profile (night 2) is restored. Feature check and a short speed check are logged.
# usage: deploy.sh DEFAULTS_JSON      e.g. deploy.sh '{"reasoning_effort":"medium"}'
set -uo pipefail
N=/root/tune/night3; NL=$N/NIGHT-LOG.md; D="$1"
PROD=/opt/vllm-qfn/qfn-prod.env; BASE=$N/qfn-prod.env.baseline
VPY=/opt/vllm-qfn-exp/.venv/bin/python
note() { echo "- $(date -u +%T) UTC [deploy] $*" >> $NL; }
health() { curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:8080/health; }
wait_health() { for i in $(seq 1 240); do [ "$(health)" = 200 ] && return 0; sleep 5; done; return 1; }
gpus_free() {
  for _ in $(seq 1 60); do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s += ($1 >= 1000)} END {exit s}' && return 0
    sleep 2
  done
}
restart_prod() { systemctl stop qfn-vllm 2>/dev/null; systemctl stop vllm-qfn; gpus_free; systemctl start vllm-qfn; wait_health; }

grep -q "Asgard patch A1" /opt/vllm-qfn-exp/src/vllm/entrypoints/openai/chat_completion/serving.py \
  || { note "vLLM patch A1 not applied - run verify.sh first"; exit 1; }
[ -s /opt/vllm-qfn/templates/qfn-asgard-v2.jinja ] || { note "template v2 missing"; exit 1; }

# new profile = baseline with the template and the server defaults swapped (python: no shell-escaping games)
python3 - "$D" "$BASE" "$PROD.new" <<'EOF' || { note "could not build the production profile"; exit 1; }
import json, re, sys
d = json.loads(sys.argv[1])
s = open(sys.argv[2]).read()
old = "--chat-template /opt/vllm-qfn/templates/chat_template-chromix-froggerick-v1.jinja"
assert s.count(old) == 1, "template flag not found"
s = s.replace(old, "--chat-template /opt/vllm-qfn/templates/qfn-asgard-v2.jinja")
m = re.search(r"--default-chat-template-kwargs (\S+)", s)
assert m, "default kwargs flag not found"
s = s[:m.start(1)] + json.dumps(d, separators=(",", ":")).replace('"', '\\"') + s[m.end(1):]
s = s.replace("# PRODUCTION profile (night 2, 2026-10-15)",
              "# PRODUCTION profile (night 3, 2026-10-17: chat template qfn-asgard-v2 + vLLM patches A1/A2; before: night 2, 2026-10-15)")
open(sys.argv[3], "w").write(s)
EOF
bash -c "set -a; source $PROD.new; set +a; echo \"\$EXTRA_FLAGS\"" | tr ' ' '\n' | grep -A1 -E "^--chat-template|^--default-chat-template-kwargs" > $N/deploy-flags.txt
note "new profile flags: $(tr '\n' ' ' < $N/deploy-flags.txt)"
grep -q "qfn-asgard-v2.jinja" $N/deploy-flags.txt && grep -q '^{' $N/deploy-flags.txt || { note "profile flags look wrong - not deploying"; exit 1; }

cp -p $PROD $N/qfn-prod.env.before-deploy
mv $PROD.new $PROD
if ! restart_prod; then
  note "8080 not healthy with the new profile -> restoring baseline"; cp $BASE $PROD; restart_prod; note "8080 now $(health)"; exit 1
fi
note "8080 healthy with qfn-asgard-v2"

ok=1
cd /root/tune
for c in prod_check func_check; do
  out=$(python3 -I $c.py 8080 2>&1); note "$c: $(echo "$out" | tail -1)"
  echo "$out" | tail -1 | grep -q "ALL PASS" || ok=0
done
out=$(python3 -I multiturn_check.py 8080 3 20000 2>&1); note "multiturn: $(echo "$out" | tail -2 | tr '\n' ' ')"
echo "$out" | tail -1 | grep -q "ALL PASS" || ok=0
if [ $ok = 0 ]; then
  note "gate FAILED -> restoring baseline profile"; cp $BASE $PROD; restart_prod; note "8080 now $(health) (baseline)"; exit 1
fi
cd /root/tpl-eval && python3 -I feature_check.py 8080 $N/feature_check-prod.json > $N/feature-prod.txt 2>&1
note "feature check on 8080: $(tail -1 $N/feature-prod.txt); fails: $(grep FAIL $N/feature-prod.txt | cut -c1-90 | tr '\n' ';')"
cd /root/tune && PERF_SIZES=4096,32768 $VPY perf_exact.py N3-prod 8080 /opt/vllm-qfn/models/swift15-awq 3 256 > $N/perf-prod.txt 2>&1
note "speed (median, vs night-2 E5): $($VPY - <<'EOF'
import json, statistics as st
def med(f):
    r = {}
    for x in json.load(open(f)):
        r.setdefault(x["size"], []).append(x)
    return {k: (st.median(y["pp"] for y in v), st.median(y["tg"] for y in v)) for k, v in r.items()}
a, b = med("/root/tune/exact-N3-prod.json"), med("/root/tune/exact-E5.json")
print("; ".join(f"{k}: pp {a[k][0]:.0f} ({a[k][0] / b[k][0]:.0%}) tg {a[k][1]:.0f} ({a[k][1] / b[k][1]:.0%})" for k in sorted(a) if k in b))
EOF
)"
note "DEPLOYED"
