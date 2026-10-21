#!/bin/bash
# Night 5 deploy: put candidate NAME into production on 8080. Its mkprof args (model / kv / spec / tplkw / head / mtp /
# pmu / venv) are re-applied to the baseline profile. Gate: prod/func/multiturn must pass, else the baseline is
# restored. Feature check + speed on 8080 are logged.  usage: deploy5.sh NAME
set -uo pipefail
N=/root/tune/night5; NL=$N/NIGHT-LOG.md; BASE=$N/qfn-prod.env.baseline; PROD=/opt/vllm-qfn/qfn-prod.env
VPY=/opt/vllm-qfn-exp/.venv/bin/python; NAME=$1
note() { echo "- $(date -u +%T) UTC [deploy] $*" >> $NL; }
health() { curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:8080/health; }
wait_health() { for _ in $(seq 1 240); do [ "$(health)" = 200 ] && return 0; sleep 5; done; return 1; }
gpus_free() {
  for _ in $(seq 1 60); do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s += ($1 >= 1000)} END {exit s}' && return 0
    sleep 2
  done; return 1
}
restart_prod() { systemctl stop qfn-vllm 2>/dev/null; systemctl stop vllm-qfn; gpus_free; systemctl start vllm-qfn; wait_health; }
rollback() { note "$1 -> restoring the baseline profile"; cp $BASE $PROD; restart_prod; note "8080 now $(health) (baseline)"; exit 1; }

args=$(sed -n 's/^# NIGHT 4 test profile .* by mkprof.py //p' $N/profiles/$NAME.env | sed 's/)$//' | tr ' ' '\n' \
       | grep -E '^(venv|head|mtp|pmu|batch|split|model|kv|spec|tplkw)=' | tr '\n' ' ')
[ -n "$args" ] || { note "no mkprof args found for $NAME"; exit 1; }
python3 $N/mkprof.py $BASE $PROD.new $args label=production-$NAME > /dev/null || { note "profile build failed"; exit 1; }
sed -i "1s|.*|# PRODUCTION profile (night 5, $(date -u +%F): $NAME = $args; before: night 4)|" $PROD.new
grep -q "^PORT=8080" $PROD.new || { note "new profile is not on 8080 - not deploying"; rm -f $PROD.new; exit 1; }
note "new production profile: $args"
cp -p $PROD $N/qfn-prod.env.before-deploy
mv $PROD.new $PROD
restart_prod || rollback "8080 not healthy with $NAME"
note "8080 healthy with $NAME ($(grep -a -m1 -o 'GPU KV cache size: [0-9,]* tokens' /var/log/vllm-qfn/vllm.log | tail -1))"

cd /root/tune
ok=1
for c in prod_check func_check; do
  out=$(python3 -I $c.py 8080 2>&1); note "$c: $(echo "$out" | tail -1)"
  echo "$out" | tail -1 | grep -q "ALL PASS" || ok=0
done
out=$(python3 -I multiturn_check.py 8080 3 20000 2>&1); note "multiturn: $(echo "$out" | tail -2 | tr '\n' ' ')"
echo "$out" | tail -1 | grep -q "ALL PASS" || ok=0
[ $ok = 1 ] || rollback "gate FAILED"
(cd /root/tpl-eval && python3 -I feature_check.py 8080 $N/feature_check-prod.json > $N/feature-prod.txt 2>&1)
note "feature check on 8080: $(tail -1 $N/feature-prod.txt); fails: $(grep FAIL $N/feature-prod.txt | cut -c1-90 | tr '\n' ';')"
PERF_SIZES=4096,32768 $VPY perf_exact.py N5-prod 8080 /opt/vllm-qfn/models/swift15-awq 3 256 > $N/perf-prod.txt 2>&1
out=$(python3 -I decode_code.py N5-prod 8080 2>&1 | grep "^|" | tail -1); note "code decode on 8080: $out"
note "speed on 8080 (median, vs night 4 production): $($VPY - <<'EOF'
import json, statistics as st
def med(f):
    r = {}
    for x in json.load(open(f)):
        r.setdefault(x["size"], []).append(x)
    return {k: (st.median(y["pp"] for y in v), st.median(y["tg"] for y in v)) for k, v in r.items()}
a, b = med("/root/tune/exact-N5-prod.json"), med("/root/tune/exact-N4-prod.json")
print("; ".join(f"{k}: pp {a[k][0]:.0f} ({a[k][0] / b[k][0]:.0%}) tg {a[k][1]:.0f} ({a[k][1] / b[k][1]:.0%})" for k in sorted(a) if k in b))
EOF
)"
note "DEPLOYED $NAME"
