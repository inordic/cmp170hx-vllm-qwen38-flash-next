# Configuration

| File | What it is |
|---|---|
| `qfn-prod.env` | the production serve profile, read by 2dameneko's `serve/serve.sh` via `RUN_ENV` |
| `vllm-qfn.service` | systemd unit inside the container: starts `serve.sh` with the profile, restarts on failure |
| `vllm-qfn.logrotate` | daily log rotation for `/var/log/vllm-qfn/` |
| `host/gpu-power-limit.service` | Proxmox host: 300 W on both cards at boot (only with the 300 W VBIOS) |
| `host/gpu-thermal-guard.sh` | Proxmox host, run every minute: lowers a card to 275 W at 90 °C core or HBM |
| `host/lxc-container.conf` | the LXC config: GPUs as device nodes, driver libraries bind-mounted, memlock unlimited |

`../scripts/qfn-window.sh` swaps the GPUs between the permanent service and a test unit (`up PROFILE` / `prod` / `status`). `../scripts/night_run.sh` runs a profile through a list of tests.
