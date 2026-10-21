#!/bin/bash
# 2026-10-21: roll production back from C1T to the night-4 standard (multi-turn tool-use drop found in night 5 data)
N=/root/tune/night5; NL=$N/NIGHT-LOG.md; PROD=/opt/vllm-qfn/qfn-prod.env
note() { echo "- $(date -u +%T) UTC [rollback] $*" >> $NL; }
health() { curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:8080/health; }
cp -p $PROD $N/qfn-prod.env.C1T
cp -p /opt/vllm-qfn/qfn-prod.env.standard-2026-10-20 $PROD
note "C1T -> night-4 standard"
systemctl restart vllm-qfn
for _ in $(seq 1 240); do [ "$(health)" = 200 ] && break; sleep 5; done
note "8080 $(health); $(grep -a "GPU KV cache size" /var/log/vllm-qfn/vllm.log | tail -1 | grep -o "GPU KV cache size: [0-9,]* tokens")"
cd /root/tune
for c in prod_check func_check; do note "$c: $(python3 -I $c.py 8080 2>&1 | tail -1)"; done
note "multiturn: $(python3 -I multiturn_check.py 8080 3 20000 2>&1 | tail -1)"
note "code decode on 8080: $(python3 -I decode_code.py N5-rollback 8080 2>&1 | grep "^|" | tail -1)"
note "rollback done"
