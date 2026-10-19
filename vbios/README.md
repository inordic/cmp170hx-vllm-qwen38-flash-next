# Cross-flashing the 250 W card to the 300 W VBIOS

> **Warning.** Flashing firmware can brick a card. We did this on our own cards at our own risk, and so would you. Keep a dump of every card's original ROM and a hardware programmer (e.g. a CH341A) at hand. **No ROM files are included here; they are NVIDIA firmware.**

## Why

Our two CMP 170HX shipped with different VBIOS versions:

| Card | VBIOS | Power | Max SM / memory clock |
|---|---|---|---|
| A | 92.00.6D.00.0A | 300 W | 1695 / 1728 MHz |
| B | 92.00.67.00.01 | 250 W | 1410 / 1458 MHz |

In a two-stage pipeline the slower card sets the pace, and card B held the pipeline back at long prefill.

After writing card A's ROM to card B (same device ID 0x20C2, subsystem 0x1585, board 0x030A):
- both cards run the 300 W VBIOS;
- code decode went from 161 to 169.5 tok/s;
- prefill gained up to 8% from 8K to 32K;
- both cards now hold about 1455–1485 MHz under load;
- 64 GiB stayed intact.

## How (Proxmox host, card not in use)

These scripts wrap NVIDIA's `nvflash` 5.867 and polyuij42-del's `check_rom.py` and `verify_170hx.sh`. Get those from their sources; they are not included here.

| Step | Script | What happens |
|---|---|---|
| 1 | `stage.sh` | saves the container's GPU device lines, removes them, and blacklists the nvidia modules for the next boot |
| 2 | reboot | the host comes up without the nvidia driver |
| 3 | `step1_dump.sh` | dumps both cards' ROMs (driverless) and runs `check_rom.py` on each. **Look at the output** before going on |
| 4 | `step2_flash.sh <index of the 250 W card> <300 W dump> <250 W dump>` | refuses unless the source ROM is 92.00.6D.00.0A with device 0x20C2 and subsystem 0x1585, and the rollback ROM is 92.00.67.00.01. Then it writes, and prints nvflash's success marker |
| 5 | `unstage.sh` | removes the blacklist and gives the container its GPU devices back. Run it **after** the driver is loaded again: `pct set --devN` fails while the device is missing |
| 6 | reboot | then `verify_170hx.sh`, and set 300 W on the flashed card ([`../config/host/gpu-power-limit.service`](../config/host/gpu-power-limit.service)) |

**Lessons:**
- A warm reboot was enough for the driverless dumps.
- `pct set` / `pct delete` on a running container stay pending until its next start, so restarting the container is enough after unstaging.
