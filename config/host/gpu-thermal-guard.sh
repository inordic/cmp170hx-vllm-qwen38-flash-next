#!/bin/bash
# Night guard: if a GPU core or HBM reaches 90 C, lower that card to 275 W (never raises anything). Log every action.
L=/root/gpu-thermal-guard.log
nvidia-smi --query-gpu=pci.bus_id,temperature.gpu,temperature.memory,power.limit --format=csv,noheader,nounits | while IFS=", " read -r bus tg tm pl; do
  m=$tg; [ "${tm:-0}" -gt "$m" ] 2>/dev/null && m=$tm
  if [ "$m" -ge 90 ] && [ "${pl%.*}" -gt 275 ]; then
    nvidia-smi -i "$bus" -pl 275 >/dev/null 2>&1 && echo "$(date -u +%FT%TZ) $bus core=${tg}C hbm=${tm}C -> power limit 275 W" >> $L
  fi
done
echo "$(date -u +%FT%TZ) $(nvidia-smi --query-gpu=index,temperature.gpu,temperature.memory,power.draw,power.limit --format=csv,noheader,nounits | tr "\n" ";")" >> /root/gpu-thermal-trace.log
