#!/bin/bash
# Back to normal: driver loads again, CT 232 gets its GPU devices back.
set -e; V=/root/vbios-flash
rm -f /etc/modprobe.d/zz-flash-window.conf
args=""; while IFS=": " read -r k v; do args="$args --$k $v"; done < $V/232.devs
pct set 232 $args
grep -E "^dev[0-9]+:" /etc/pve/lxc/232.conf
echo "UNSTAGED: blacklist removed, CT 232 GPU devices restored. Reboot next."
