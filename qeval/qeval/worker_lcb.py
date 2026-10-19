"""LiveCodeBench grader. Runs INSIDE the sandbox (as nobody). Stdlib only.

Grading logic adapted from LiveCodeBench lcb_runner/evaluation/testing_util.py (MIT License,
Copyright (c) LiveCodeBench authors): same import prelude, same stdin mocking, same
`wrapped_function` transform, same per-test timeout (default 6 s), same line-wise comparison
with Decimal fallback, same call-based comparison (tuple -> list, ==). Stops at first failure.
"""
import ast
import faulthandler
import json
import os
import resource
import signal
import sys
import time
from decimal import Decimal
from io import StringIO
from types import ModuleType
from unittest.mock import mock_open, patch

MARK = "@@QEVAL_RESULT@@"
_REAL_OUT = os.fdopen(os.dup(1), "w")

import_string = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"


def emit(d):
    _REAL_OUT.write("\n" + MARK + json.dumps(d) + "\n")
    _REAL_OUT.flush()


class TimeoutException(BaseException):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def truncatefn(s, length=300):
    s = s if isinstance(s, str) else str(s)
    return s if len(s) <= length else s[: length // 2] + "...(truncated) ..." + s[-length // 2:]


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        self._stringio.close = lambda x=None: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


class MockBuffer:
    def __init__(self, inputs):
        self.inputs = inputs.encode("utf-8")

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self.inputs.split(b"\n")[0] + b"\n"


class MockStdinWithBuffer:
    def __init__(self, inputs):
        self.inputs = inputs
        self._stringio = StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __getattr__(self, name):
        return getattr(self._stringio, name)


def clean_if_name(code):
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            if ast.unparse(last_block.test).strip() == "__name__ == '__main__'":
                code = ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)
    except Exception:
        pass
    return code


def make_function(code):
    try:
        import_stmts, others = [], []
        astree = ast.parse(code)
        for stmt in astree.body:
            (import_stmts if isinstance(stmt, (ast.Import, ast.ImportFrom)) else others).append(stmt)
        fn = ast.FunctionDef(name="wrapped_function",
                             args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                             body=others, decorator_list=[], lineno=-1)
        return import_string + "\n" + ast.unparse(import_stmts) + "\n" + ast.unparse(fn)
    except Exception:
        return code


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)
    it = iter(inputs.split("\n"))
    mock_stdin = MockStdinWithBuffer(inputs)

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", mock_stdin)
    @patch("sys.stdin.readline", lambda *args: next(it))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner(_method):
        try:
            return _method()
        except SystemExit:
            pass

    return _inner(method)


def compile_code(code, timeout):
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        compiled = tmp_sol.Solution() if "class Solution" in code else tmp_sol
    finally:
        signal.alarm(0)
    return compiled


def convert_line_to_decimals(line):
    try:
        return True, [Decimal(e) for e in line.split()]
    except Exception:
        return False, []


def get_stripped_lines(val):
    return [l.strip() for l in val.strip().split("\n")]


def grade_call_based(code, all_inputs, all_outputs, fn_name, timeout):
    code = import_string + "\n\n" + code
    compiled = compile_code(code, timeout)
    method = getattr(compiled, fn_name, None)
    if method is None:
        return [-4], {"error_code": -4, "error_message": f"function {fn_name} not found"}
    all_inputs = [[json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs]
    all_outputs = [json.loads(o) for o in all_outputs]
    results = []
    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            pred = method(*gt_inp)
            signal.alarm(0)
            if isinstance(pred, tuple):
                pred = list(pred)
            ok = pred == gt_out
            results.append(ok)
            if not ok:
                return results, {"error_code": -2, "error_message": "Wrong Answer", "output": truncatefn(pred),
                                 "expected": truncatefn(gt_out)}
        except TimeoutException:
            results.append(-3)
            return results, {"error_code": -3, "error_message": "Time Limit Exceeded"}
        except BaseException as e:
            signal.alarm(0)
            results.append(-4)
            return results, {"error_code": -4, "error_message": "Runtime Error", "error": truncatefn(repr(e))}
        finally:
            signal.alarm(0)
            faulthandler.disable()
    return results, {}


def grade_stdio(code, all_inputs, all_outputs, timeout):
    code = make_function(clean_if_name(code))
    compiled = compile_code(code, timeout)
    method = getattr(compiled, "wrapped_function", None)
    if method is None:
        return [-4], {"error_code": -4, "error_message": "no wrapped_function"}
    results = []
    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        with Capturing() as captured:
            try:
                call_method(method, gt_inp)
                signal.alarm(0)
            except TimeoutException:
                signal.alarm(0)
                results.append(-3)
                return results, {"error_code": -3, "error_message": "Time Limit Exceeded"}
            except BaseException as e:
                signal.alarm(0)
                results.append(-4)
                return results, {"error_code": -4, "error_message": "Runtime Error", "error": truncatefn(repr(e))}
            finally:
                signal.alarm(0)
                faulthandler.disable()
        pred = captured[0]
        pl, gl = get_stripped_lines(pred), get_stripped_lines(gt_out)
        wa = {"error_code": -2, "output": truncatefn(pred), "expected": truncatefn(gt_out)}
        if len(pl) != len(gl):
            results.append(-2)
            wa["error_message"] = "Wrong answer: mismatched output length"
            return results, wa
        for i, (a, b) in enumerate(zip(pl, gl)):
            if a == b:
                continue
            ok1, da = convert_line_to_decimals(a)
            ok2, db = convert_line_to_decimals(b)
            if ok1 and ok2 and da == db:
                continue
            results.append(-2)
            wa["error_message"] = f"Wrong answer at line {i}"
            return results, wa
        results.append(True)
    return results, {}


def reliability_guard(mem):
    if mem:
        for name in ("RLIMIT_AS", "RLIMIT_DATA"):
            try:
                resource.setrlimit(getattr(resource, name), (mem, mem))
            except Exception:
                pass
    for name, val in (("RLIMIT_FSIZE", 64 << 20), ("RLIMIT_CORE", 0)):
        try:
            resource.setrlimit(getattr(resource, name), (val, val))
        except Exception:
            pass
    if os.getuid() != 0:
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (512, 512))  # threads count too on Linux
        except Exception:
            pass
    faulthandler.disable()
    import builtins
    builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"
    for n in ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid", "fork", "forkpty",
              "killpg", "rename", "renames", "truncate", "replace", "unlink", "fchmod", "fchown", "chmod", "chown",
              "chroot", "lchflags", "lchmod", "lchown", "getcwd", "chdir"):
        if hasattr(os, n):
            setattr(os, n, None)
    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    import subprocess
    subprocess.Popen = None
    for m in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[m] = None


def main():
    p = json.loads(sys.stdin.buffer.read())
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    lim = p.get("_limits", {})
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (lim.get("cpu", 600), lim.get("cpu", 600)))
    except Exception:
        pass
    timeout = int(p.get("timeout", 6))
    tests = p["tests"]
    signal.signal(signal.SIGALRM, timeout_handler)
    reliability_guard(lim.get("mem"))
    t0 = time.time()
    try:
        if tests.get("fn_name"):
            results, meta = grade_call_based(p["code"], tests["inputs"], tests["outputs"], tests["fn_name"], timeout)
        else:
            results, meta = grade_stdio(p["code"], tests["inputs"], tests["outputs"], timeout)
    except BaseException as e:
        results, meta = [-4], {"error_code": -4, "error_message": f"Error during testing: {truncatefn(repr(e))}"}
    n = len(tests["inputs"])
    passed = len(results) == n and all(r is True for r in results)
    emit({"passed": passed, "n_tests": n, "n_run": len(results), "meta": meta, "exec_s": round(time.time() - t0, 3)})


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        emit({"passed": False, "meta": {"error_code": -5, "error_message": f"worker: {type(e).__name__}: {e}"}})
