#!/bin/bash
# Enter flash mode: next boot loads NO nvidia driver; CT 232 starts without GPU devices (jump host only).
set -e; V=/root/vbios-flash
[ -f $V/232.conf.backup ] || cp /etc/pve/lxc/232.conf $V/232.conf.backup
grep -E "^dev[0-9]+:" $V/232.conf.backup > $V/232.devs
pct set 232 --delete dev0,dev1,dev2,dev3,dev4
printf "install nvidia /bin/false\ninstall nvidia_modeset /bin/false\ninstall nvidia_drm /bin/false\ninstall nvidia_uvm /bin/false\n" > /etc/modprobe.d/zz-flash-window.conf
echo "STAGED: blacklist in place, CT 232 GPU devices removed (saved in $V/232.devs). Reboot next."
