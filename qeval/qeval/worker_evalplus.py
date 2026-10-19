"""EvalPlus-style differential tester. Runs INSIDE the sandbox (as nobody). Stdlib only.

Reads one JSON payload from stdin, prints MARK + JSON result. Semantics follow
evalplus 0.3.1 (evalplus/eval/__init__.py, evaluate.py; Apache-2.0):
expected outputs come from the canonical solution in this same process; per-input time limit
max(1.0 s, 4 x canonical time); exact match, else allclose(rtol=1e-7, atol) with atol 1e-6
forced for float outputs; special oracles for a few HumanEval/MBPP tasks.
"""
import ast
import contextlib
import copy
import io
import json
import math
import os
import resource
import signal
import sys
import time

MARK = "@@QEVAL_RESULT@@"
_REAL_OUT = os.fdopen(os.dup(1), "w")


class _Timeout(BaseException):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def emit(d):
    _REAL_OUT.write("\n" + MARK + json.dumps(d) + "\n")
    _REAL_OUT.flush()


def limits(lim):
    for name, val in (("RLIMIT_AS", lim.get("mem")), ("RLIMIT_CPU", lim.get("cpu")),
                      ("RLIMIT_FSIZE", 16 << 20), ("RLIMIT_CORE", 0)):
        if val is None or not hasattr(resource, name):
            continue
        try:
            resource.setrlimit(getattr(resource, name), (val, val))
        except Exception:
            pass
    if os.getuid() != 0:
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (512, 512))  # threads count too on Linux
        except Exception:
            pass


# ---- special oracles (verbatim logic from evalplus/eval/_special_oracle.py) ----
MBPP_OUTPUT_NOT_NONE_TASKS = ["check_str", "text_match_three", "text_starta_endb"]
MBPP_OUTPUT_SET_EQ_TASKS = ["similar_elements", "find_char_long", "common_in_nested_lists",
                            "extract_singly", "larg_nnum", "intersection_array", "find_dissimilar", "Diff"]


def _surface_Area(base_edge, height):
    slant_height = math.sqrt((base_edge / 2) ** 2 + height ** 2)
    base_area = base_edge ** 2
    lateral_area = 4 * (base_edge * slant_height) / 2
    return round(base_area + lateral_area)


def _digit_distance_nums(num1, num2):
    s1, s2 = str(num1), str(num2)
    m = max(len(s1), len(s2))
    s1, s2 = s1.zfill(m), s2.zfill(m)
    return sum(abs(int(a) - int(b)) for a, b in zip(s1, s2))


def _poly(xs, x):
    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])


# ---- verbatim from evalplus/data/mbpp.py (Apache-2.0): JSON lists -> tuples/sets/complex ----
def mbpp_deserialize_inputs(task_id: str, inputs: list) -> list:
    task_id = int(task_id.split("/")[-1])
    if task_id in [
        2,
        116,
        132,
        143,
        222,
        261,
        273,
        394,
        399,
        421,
        424,
        429,
        470,
        560,
        579,
        596,
        616,
        630,
        726,
        740,
        744,
        809,
    ]:
        modified_inputs = [[tuple(lst) for lst in inp] for inp in inputs]

    elif task_id in [
        63,
        64,
        70,
        94,
        120,
        237,
        272,
        299,
        400,
        409,
        417,
        438,
        473,
        614,
        780,
    ]:
        modified_inputs = [
            [[tuple(lst) for lst in lst_lst] for lst_lst in inp] for inp in inputs
        ]

    elif task_id in [75, 413, 444, 753]:
        modified_inputs = [
            [[tuple(lst) for lst in inp[0]]] + [inp[1]] for inp in inputs
        ]

    elif task_id == 106 or task_id == 750:
        modified_inputs = [[inp[0]] + [tuple(inp[1])] for inp in inputs]

    elif task_id == 115:
        modified_inputs = [
            [
                [
                    set(item) if isinstance(item, list) and len(item) else {}
                    for item in inp[0]
                ]
            ]
            for inp in inputs
        ]

    elif task_id == 124:
        modified_inputs = [(float(inp[0]), complex(inp[1])) for inp in inputs]

    elif task_id in [250, 405, 446, 617, 720, 763, 808]:
        modified_inputs = [[tuple(inp[0])] + [inp[1]] for inp in inputs]

    elif task_id in [259, 401, 445]:
        modified_inputs = [
            [[tuple(lst) for lst in lst_lst] for lst_lst in inp] for inp in inputs
        ]
        modified_inputs = [[tuple(lst) for lst in inp] for inp in modified_inputs]

    elif task_id == 278:
        modified_inputs = [
            [[tuple(item) if isinstance(item, list) else item for item in inp[0]]]
            for inp in inputs
        ]
        modified_inputs = [[tuple(lst) for lst in inp] for inp in modified_inputs]

    elif task_id == 307:
        modified_inputs = [[tuple(inp[0])] + [inp[1], inp[2]] for inp in inputs]

    elif task_id == 722:
        modified_inputs = [
            [{key: tuple(value) for key, value in inp[0].items()}] + inp[1:]
            for inp in inputs
        ]

    elif task_id == 252:
        modified_inputs = [[complex(inp[0])] for inp in inputs]

    elif task_id in [580, 615, 791]:

        def turn_all_list_into_tuple(inp):
            if isinstance(inp, list):
                return tuple([turn_all_list_into_tuple(item) for item in inp])
            return inp

        modified_inputs = [turn_all_list_into_tuple(inp) for inp in inputs]

    else:
        modified_inputs = inputs

    return modified_inputs


def is_floats(x):
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)) and x:
        return all(isinstance(i, float) for i in x)
    return False


def _flat(x):
    if isinstance(x, (list, tuple)):
        out = []
        for i in x:
            out.append(_flat(i))
        return out
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, complex):
        return x
    raise TypeError("non-numeric")


def allclose(a, b, rtol, atol):
    """Pure-python stand-in for np.allclose on scalars / (nested) equal-shape sequences."""
    a, b = _flat(a), _flat(b)

    def rec(x, y):
        if isinstance(x, list) or isinstance(y, list):
            if not (isinstance(x, list) and isinstance(y, list)):
                # numpy broadcasting of a scalar against a list
                if isinstance(x, list):
                    return all(rec(i, y) for i in x)
                return all(rec(x, j) for j in y)
            if len(x) != len(y):
                if len(x) == 1:
                    return all(rec(x[0], j) for j in y)
                if len(y) == 1:
                    return all(rec(i, y[0]) for i in x)
                raise ValueError("shape mismatch")
            return all(rec(i, j) for i, j in zip(x, y))
        if isinstance(x, float) and isinstance(y, float):
            if math.isinf(x) or math.isinf(y):
                return x == y
            if math.isnan(x) or math.isnan(y):
                return False
        return abs(x - y) <= atol + rtol * abs(y)

    return rec(a, b)


def sanitize(code, entry_point):
    """Keep imports, defs, classes and simple top-level assignments (like evalplus.sanitize)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            keep.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            keep.append(node)
    mod = ast.Module(body=keep, type_ignores=[])
    try:
        return ast.unparse(mod)
    except Exception:
        return code


def run_fn(fn, inp, tl):
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, tl)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return fn(*inp)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def main():
    p = json.loads(sys.stdin.buffer.read())
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    limits(p.get("_limits", {}))
    dataset, ep, atol0 = p["dataset"], p["entry_point"], p.get("atol", 0) or 0
    min_tl, factor = p.get("min_time_limit", 1.0), p.get("gt_time_limit_factor", 4.0)
    per_task_cap = p.get("timeout_per_task", 60)
    not_none = ep in MBPP_OUTPUT_NOT_NONE_TASKS if dataset == "mbpp" else False
    if dataset == "mbpp":
        p["base_input"] = mbpp_deserialize_inputs(p["task_id"], p["base_input"])
        p["plus_input"] = mbpp_deserialize_inputs(p["task_id"], p["plus_input"])

    # ---- ground truth (canonical solution) ----
    g = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(p["canonical"], "<canonical>", "exec"), g)
    gfn = g[ep]
    phases = {}
    for ph in ("base", "plus"):
        exp, tm = [], []
        for inp in p[ph + "_input"]:
            inp2 = copy.deepcopy(inp)
            t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                out = gfn(*inp2)
            tm.append(time.time() - t0)
            exp.append((out is not None) if not_none else out)
        phases[ph] = (exp, tm)

    # ---- candidate ----
    code = sanitize(p["solution"], ep)
    res = {"base": "fail", "plus": "fail", "fail_phase": None, "fail_index": None, "error": None}
    ns = {}
    try:
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, 10.0)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exec(compile(code, "<solution>", "exec"), ns)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        fn = ns[ep]
    except BaseException as e:
        res["error"] = f"load: {type(e).__name__}: {str(e)[:300]}"
        res["fail_phase"] = "load"
        emit(res)
        return
    for ph in ("base", "plus"):
        exp_list, ref_t = phases[ph]
        inputs = p[ph + "_input"]
        tls = [max(min_tl, factor * t) for t in ref_t]
        budget = min(per_task_cap, sum(tls)) + 1
        t_phase = time.time()
        status = "pass"
        for i, inp in enumerate(inputs):
            if time.time() - t_phase > budget:
                status = "timeout"
                res["fail_index"] = i
                break
            atol = atol0
            try:
                out = run_fn(fn, copy.deepcopy(inp), tls[i])
                exp = exp_list[i]
                exact = out == exp
                if dataset == "mbpp":
                    if ep == "are_equivalent":
                        exact = exact or True
                    elif ep == "sum_div":
                        exact = exact or out == 0
                    elif ep == "surface_Area":
                        exact = exact or abs(out - _surface_Area(*inp)) <= atol
                    elif ep == "digit_distance_nums":
                        exact = exact or out == _digit_distance_nums(*inp)
                    elif ep in MBPP_OUTPUT_SET_EQ_TASKS:
                        exact = set(out) == set(exp)
                    elif ep in MBPP_OUTPUT_NOT_NONE_TASKS:
                        exact = (out == exp) if isinstance(out, bool) else (exp == (out is not None))
                if dataset == "humaneval" and ep == "find_zero":
                    assert abs(_poly(*inp, out)) <= atol
                    continue
                if atol == 0 and is_floats(exp):
                    atol = 1e-6
                if not exact and atol != 0:
                    assert type(out) == type(exp)
                    if isinstance(exp, (list, tuple)):
                        assert len(out) == len(exp)
                    assert allclose(out, exp, 1e-07, atol)
                else:
                    assert exact
            except _Timeout:
                status = "timeout"
                res["fail_index"] = i
                res["error"] = "per-input time limit"
                break
            except BaseException as e:
                status = "fail"
                res["fail_index"] = i
                res["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                break
        res[ph] = status
        if status != "pass":
            res["fail_phase"] = ph
            break
    emit(res)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:  # never die silently
        emit({"base": "fail", "plus": "fail", "fail_phase": "worker", "error": f"{type(e).__name__}: {e}"})
