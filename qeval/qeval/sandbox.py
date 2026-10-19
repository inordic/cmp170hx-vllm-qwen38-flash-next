"""Run model-written code as an unprivileged user with resource limits and a hard timeout.

On Linux as root: each run is `unshare -n` (no network, if the kernel/container allows it) +
`setpriv` (or Popen(user=...)) to nobody:nogroup, in a fresh temp dir owned by nobody,
process-group kill on timeout. The worker script sets RLIMIT_AS/CPU/FSIZE/NPROC itself before
it touches the untrusted code, so no preexec_fn is needed (thread-safe).
Not root: runs as the current user with the same rlimits (prints a warning once).
"""
from __future__ import annotations

import json
import os
import pwd
import grp
import shutil
import signal
import subprocess
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
MARK = "@@QEVAL_RESULT@@"
_lock = threading.Lock()
_state = {}


def _pick_python(explicit=None):
    cands = [explicit] if explicit else []
    cands += [sys.executable, "/usr/bin/python3.12", "/usr/bin/python3"]
    for c in cands:
        if c and os.path.exists(c):
            # nobody must be able to run it: reject interpreters inside /root or other 0700 trees
            p = os.path.realpath(c)
            ok = True
            d = os.path.dirname(p)
            while d and d != "/":
                st = os.stat(d)
                if not (st.st_mode & 0o001):
                    ok = False
                    break
                d = os.path.dirname(d)
            if ok or os.geteuid() != 0:
                return c
    return sys.executable


def setup(sandbox_python=None, base_dir=None, exec_workers=3):
    """Idempotent. Copies worker scripts to a world-readable dir and probes isolation."""
    with _lock:
        if _state:
            return _state
        base = base_dir or os.path.join(tempfile.gettempdir(), f"qeval-sbx-{os.getpid()}")
        os.makedirs(base, exist_ok=True)
        os.chmod(base, 0o711)
        for w in ("worker_evalplus.py", "worker_lcb.py"):
            dst = os.path.join(base, w)
            shutil.copyfile(os.path.join(HERE, w), dst)
            os.chmod(dst, 0o644)
        root = os.geteuid() == 0
        mode = "self"
        prefix = []
        uid = gid = None
        if sys.platform.startswith("linux") and root:
            try:
                uid = pwd.getpwnam("nobody").pw_uid
                try:
                    gid = grp.getgrnam("nogroup").gr_gid
                except KeyError:
                    gid = grp.getgrnam("nobody").gr_gid
                mode = "nobody"
            except KeyError:
                pass
            if mode == "nobody":
                have_unshare = shutil.which("unshare") is not None
                have_setpriv = shutil.which("setpriv") is not None
                netns = False
                if have_unshare:
                    try:
                        netns = subprocess.run(["unshare", "-n", "true"], timeout=10,
                                               capture_output=True).returncode == 0
                    except Exception:
                        netns = False
                if have_setpriv:
                    prefix = (["unshare", "-n", "--"] if netns else []) + [
                        "setpriv", f"--reuid={uid}", f"--regid={gid}", "--clear-groups",
                        "--no-new-privs", "--"]
                    mode = "nobody+netns" if netns else "nobody"
                else:
                    prefix = []  # fall back to Popen(user=, group=)
        if sys.platform == "darwin" and shutil.which("sandbox-exec"):
            # local testing on a Mac: no network, writes only inside the sandbox base dir
            rb = os.path.realpath(base)
            prof = ('(version 1)(allow default)(deny network*)(deny file-write*)'
                    f'(allow file-write* (subpath "{rb}"))(allow file-write* (literal "/dev/null"))')
            prefix = ["sandbox-exec", "-p", prof]
            mode = "self+sandbox-exec"
        py = _pick_python(sandbox_python)
        _state.update(dict(base=base, mode=mode, prefix=prefix, uid=uid, gid=gid, python=py,
                           sem=threading.Semaphore(max(1, exec_workers))))
        if mode.startswith("self"):
            print(f"[sandbox] WARNING: not running as root on Linux -> code runs as the current user "
                  f"(rlimits + timeout only). base={base}", file=sys.stderr)
        else:
            print(f"[sandbox] mode={mode} python={py} base={base}", file=sys.stderr)
        return _state


def run_worker(worker: str, payload: dict, timeout: float, mem_bytes: int = 4 << 30) -> dict:
    st = setup()
    payload = dict(payload)
    payload["_limits"] = {"mem": mem_bytes, "cpu": int(timeout) + 5}
    data = json.dumps(payload).encode()
    with st["sem"]:
        wd = tempfile.mkdtemp(prefix="run-", dir=st["base"])
        try:
            if st["uid"] is not None:
                os.chown(wd, st["uid"], st["gid"])
            os.chmod(wd, 0o700)
            cmd = st["prefix"] + [st["python"], "-I", "-B", os.path.join(st["base"], worker)]
            kw = {}
            if st["mode"].startswith("nobody") and not st["prefix"]:
                kw = dict(user=st["uid"], group=st["gid"], extra_groups=[])
            env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": wd, "TMPDIR": wd,
                   "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, cwd=wd, env=env, start_new_session=True, **kw)
            try:
                out, err = p.communicate(data, timeout=timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except Exception:
                    p.kill()
                out, err = p.communicate()
                timed_out = True
        finally:
            shutil.rmtree(wd, ignore_errors=True)
    txt = out.decode("utf-8", "replace")
    i = txt.rfind(MARK)
    if i >= 0:
        try:
            res = json.loads(txt[i + len(MARK):].strip().splitlines()[0])
            res["sandbox_timeout"] = timed_out
            return res
        except Exception:
            pass
    return {"passed": False, "status": "timeout" if timed_out else "crash", "sandbox_timeout": timed_out,
            "error": (err.decode("utf-8", "replace")[-800:] if err else "") or f"rc={p.returncode}"}
