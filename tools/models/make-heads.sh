#!/bin/bash
# int4 MTP draft heads (AlexBrandes tools/quantize_lm_head.py, MIT) for Swift 1.5 AWQ and Intel AutoRound
export OMP_NUM_THREADS=4
for m in swift15-awq intel-autoround; do
  echo "=== $m $(date -u +%T)"
  /opt/vllm-qfn-n4/.venv/bin/python /opt/vllm-qfn-n4/quantize_lm_head.py --model-dir /opt/vllm-qfn/models/$m     --out /opt/vllm-qfn/models/draft-heads/$m-lm_head-w4a16.safetensors --mse-scale-search
done
echo "HEADS-DONE $(date -u +%T)"
