#!/bin/bash
# Flash boot 1 (driverless): save BOTH cards VBIOS - each save is the first Falcon op on its card.
V=/root/vbios-flash; NV=$V/nvflash; L=$V/logs; exec > >(tee -a $L/step1.log) 2>&1
echo "=== step1 $(date -u)"; lsmod | grep -q "^nvidia" && { echo "ABORT: nvidia driver is loaded"; exit 1; }
systemctl stop systemd-udevd-control.socket systemd-udevd-kernel.socket systemd-udevd-varlink.socket systemd-udevd.service nvidia-persistenced 2>/dev/null
for i in 0 1; do
  echo "--- save adapter $i"
  ( sleep 45; echo y; sleep 60 ) | timeout 240 script -qec "$NV -i $i --save $V/roms/idx$i.rom" $L/save_idx$i.ts >/dev/null 2>&1
  echo "rc=$?"; tr -d "\r" < $L/save_idx$i.ts | grep -v "^y$" | tail -6
done
systemctl start systemd-udevd.service systemd-udevd-control.socket systemd-udevd-kernel.socket systemd-udevd-varlink.socket 2>/dev/null
for i in 0 1; do
  f=$V/roms/idx$i.rom; echo "=== idx$i: $(stat -c %s $f 2>/dev/null) bytes, md5 $(md5sum $f 2>/dev/null | cut -c1-32)"
  $NV --version $f 2>&1 | grep -E "Version|Device ID|Subsystem ID|Board ID|Hierarchy|Build" | head -8
  python3 $V/upstream/check_rom.py $f 2>&1 | tail -12
done
echo "=== step1 done"
