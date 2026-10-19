#!/bin/bash
# Flash boot 2 (driverless): write the 300 W ROM (dumped from the 300 W card) to the 250 W card.
# usage: step2_flash.sh TARGET_INDEX SOURCE_ROM ROLLBACK_ROM   (values come from the step1 output)
V=/root/vbios-flash; NV=$V/nvflash; L=$V/logs; IDX=$1; ROM=$2; RB=$3; exec > >(tee -a $L/step2.log) 2>&1
echo "=== step2 $(date -u) target idx $IDX rom $ROM rollback $RB"
die() { echo "FATAL: $*"; exit 1; }
[ -n "$IDX" ] && [ -f "$ROM" ] && [ -f "$RB" ] || die "usage: step2_flash.sh IDX ROM ROLLBACK"
lsmod | grep -q "^nvidia" && die "nvidia driver is loaded"
$NV --version $ROM 2>&1 | grep -q "92.00.6D.00.0A" || die "source ROM is not 92.00.6D.00.0A"
$NV --version $ROM 2>&1 | grep -q "Device ID *: 0x20C2" || die "source ROM device id != 0x20C2"
$NV --version $ROM 2>&1 | grep -q "Subsystem ID *: 0x1585" || die "source ROM subsystem != 0x1585"
$NV --version $RB 2>&1 | grep -q "92.00.67.00.01" || die "rollback ROM is not the 250 W 92.00.67.00.01"
systemctl stop systemd-udevd-control.socket systemd-udevd-kernel.socket systemd-udevd-varlink.socket systemd-udevd.service nvidia-persistenced 2>/dev/null
echo "--- WRITE (first Falcon op on adapter $IDX this boot, timed y via pty)"
( sleep 45; echo y; sleep 500 ) | timeout 600 script -qec "$NV -i $IDX $ROM" $L/flash_idx$IDX.ts >/dev/null 2>&1
echo "FLASH-RC=$?"; tr -d "\r" < $L/flash_idx$IDX.ts | grep -vE "^y$|^$" | tail -25
systemctl start systemd-udevd.service systemd-udevd-control.socket systemd-udevd-kernel.socket systemd-udevd-varlink.socket 2>/dev/null
tr -d "\r" < $L/flash_idx$IDX.ts | grep -q "A reboot is required" && echo "=== SUCCESS MARKER FOUND" || echo "=== NO SUCCESS MARKER - do not unstage yet"
