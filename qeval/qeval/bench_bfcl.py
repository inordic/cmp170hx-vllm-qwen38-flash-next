"""BFCL v4 (Berkeley Function Calling Leaderboard) in native tool-calling (FC) mode.

Uses the server's own tool parser (vLLM qwen3_xml / qwen3_coder) via `tools=[...]`, exactly
the path an agentic client uses. Scoring uses BFCL's AST checker (single-turn) and BFCL's
state + response checker (multi-turn), vendored in bfcl_vendor/ (Apache-2.0).
Data: data/bfcl/BFCL_v4_<category>.json (+ possible_answer/, multi_turn_func_doc/).
"""
import copy
import json
import os
import re

from .bfcl_vendor import ast_checker as AC
from .bfcl_vendor import multi_turn as MT
from .common import DATA, read_jsonl, subsample

SINGLE_TURN = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance",
               "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
               "live_irrelevance", "live_relevance"]
MULTI_TURN = ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"]
LANG_HINT = " Note that the provided function is in Python 3 syntax."
GORILLA_TO_OPENAPI = {
    "integer": "integer", "number": "number", "float": "number", "string": "string", "boolean": "boolean",
    "bool": "boolean", "array": "array", "list": "array", "dict": "object", "object": "object",
    "tuple": "array", "any": "string", "byte": "integer", "short": "integer", "long": "integer",
    "double": "number", "char": "string", "ArrayList": "array", "Array": "array", "HashMap": "object",
    "Hashtable": "object", "Queue": "array", "Stack": "array", "Any": "string", "String": "string",
    "Bigint": "integer",
}


def _bfcl_dir():
    return os.path.join(DATA, "bfcl")


def _load_cat(cat):
    d = _bfcl_dir()
    entries = read_jsonl(os.path.join(d, f"BFCL_v4_{cat}.json"))
    pa_path = os.path.join(d, "possible_answer", f"BFCL_v4_{cat}.json")
    answers = {a["id"]: a["ground_truth"] for a in read_jsonl(pa_path)} if os.path.exists(pa_path) else {}
    return entries, answers


def _func_docs_for(classes):
    out = []
    for c in classes:
        out.extend(read_jsonl(os.path.join(_bfcl_dir(), "multi_turn_func_doc", MT.MULTI_TURN_FUNC_DOC_FILE_MAPPING[c])))
    return out


def _add_hint(functions):
    for f in functions:
        f["description"] = f.get("description", "") + LANG_HINT
    return functions


def load(spec):
    """spec: {category: n or None (all)}; deterministic subsample per category."""
    items = []
    for cat, n in spec.items():
        entries, answers = _load_cat(cat)
        if cat.startswith("multi_turn"):
            for e in entries:
                e["function"] = _func_docs_for(e["involved_classes"])
                if "missed_function" in e:
                    for t, names in list(e["missed_function"].items()):
                        held = []
                        for name in names:
                            for i, fd in enumerate(e["function"]):
                                if fd["name"] == name:
                                    held.append(fd)
                                    e["function"].pop(i)
                                    break
                        e["missed_function"][t] = held
        for e in entries:
            # upstream adds the hint after the held-out docs were removed, so they get none
            e["function"] = _add_hint(e.get("function", []))
        chosen = subsample(entries, n, seed="bfcl-" + cat)
        for e in chosen:
            items.append({"id": e["id"], "subset": cat, "category": cat, "entry": e,
                          "answer": answers.get(e["id"])})
    return items


def _cast(properties):
    for key, value in properties.items():
        if "type" not in value:
            value["type"] = "string"
        else:
            t = value["type"]
            if t == "float":
                value["format"] = "float"
                value["description"] = value.get("description", "") + " This is a float type value."
            value["type"] = GORILLA_TO_OPENAPI.get(t, "string")
        if value["type"] in ("array", "object"):
            if "properties" in value:
                value["properties"] = _cast(value["properties"])
            elif "items" in value:
                it = value["items"]
                it["type"] = GORILLA_TO_OPENAPI.get(it.get("type", "string"), "string")
                if it["type"] == "array" and "items" in it:
                    it["items"]["type"] = GORILLA_TO_OPENAPI.get(it["items"].get("type", "string"), "string")
                elif it["type"] == "object" and "properties" in it:
                    it["properties"] = _cast(it["properties"])
    return properties


def to_tools(functions):
    tools = []
    for f in copy.deepcopy(functions):
        f["name"] = re.sub(r"\.", "_", f["name"])
        params = f.get("parameters") or {"type": "dict", "properties": {}}
        params["type"] = "object"
        params["properties"] = _cast(params.get("properties", {}))
        f["parameters"] = params
        if "response" in f:  # OpenAI-style FC handlers drop/ignore the response schema
            del f["response"]
        tools.append({"type": "function", "function": f})
    return tools


def _decode(tool_calls):
    """[{name: params_dict}] like BFCL's FC decode_ast; raises on bad JSON."""
    out = []
    for tc in tool_calls:
        args = tc.get("arguments") or "{}"
        params = json.loads(args) if isinstance(args, str) else args
        if not isinstance(params, dict):
            raise ValueError("arguments are not an object")
        out.append({tc["name"]: params})
    return out


def _is_fc_format(decoded):
    if type(decoded) != list:
        return False
    for item in decoded:
        if type(item) != dict or len(item) != 1 or type(list(item.values())[0]) != dict:
            return False
    return True


def _raw_markup(r):
    t = (r.get("content") or "")
    return ("<tool_call>" in t) or ("<function=" in t)


def _assistant_msg(r, cfg):
    m = {"role": "assistant", "content": r["content"] or ""}
    if r["tool_calls"]:
        m["tool_calls"] = [{"id": tc.get("id") or f"call_{i}", "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"] or "{}"}}
                           for i, tc in enumerate(r["tool_calls"])]
    if r.get("reasoning") and cfg.history_reasoning != "none":
        if cfg.history_reasoning in ("reasoning_content", "both"):
            m["reasoning_content"] = r["reasoning"]
        if cfg.history_reasoning in ("reasoning", "both"):
            m["reasoning"] = r["reasoning"]
    return m


def make_fn(cfg):
    def single(item, ctx):
        cat, e = item["category"], item["entry"]
        tools = to_tools(e["function"])
        msgs = e["question"][0]
        r = ctx.chat(msgs, tools=tools, max_tokens=cfg.max_tokens)
        if r.get("error"):
            return {"correct": False, "status": "api_error"}
        try:
            decoded = _decode(r["tool_calls"]) if r["tool_calls"] else None
            decode_err = None if decoded else "no tool call"
        except Exception as ex:
            decoded, decode_err = None, f"decode: {ex}"
        rec = {"n_tool_calls": len(r["tool_calls"]),
               # tool-call markup left in the text = the server's tool parser did not pick it up
               "raw_tool_markup": _raw_markup(r),
               "tool_calls": [{"name": t["name"], "arguments": t["arguments"][:1500]} for t in r["tool_calls"]],
               "text": (r["content"] or "")[-800:]}
        if "irrelevance" in cat:
            ok = not decoded
            return dict(rec, correct=ok, status="pass" if ok else "called_function")
        if cat == "live_relevance":
            ok = bool(decoded)
            return dict(rec, correct=ok, status="pass" if ok else (decode_err or "no_call"))
        if not decoded:
            return dict(rec, correct=False, status=decode_err or "no_call")
        if not _is_fc_format(decoded):
            return dict(rec, correct=False, status="wrong_output_format")
        res = AC.ast_checker(e["function"], decoded, item["answer"], AC.Language.PYTHON, cat, "qeval")
        return dict(rec, correct=bool(res["valid"]), status="pass" if res["valid"] else res.get("error_type"),
                    checker_error=None if res["valid"] else str(res.get("error"))[:600])

    def multi(item, ctx):
        cat, e = item["category"], copy.deepcopy(item["entry"])
        gt = item["answer"]
        long_ctx = "long_context" in cat
        ex = MT.Executor(e.get("initial_config", {}), e["involved_classes"], long_ctx)
        holdout = e.get("missed_function", {}) or {}
        functions = e["function"]
        tools = to_tools(functions)
        messages, turns, force_quit, api_err = [], [], False, None
        steps_total, raw_markup = 0, 0
        for t_idx, turn_msgs in enumerate(e["question"]):
            if str(t_idx) in holdout:
                functions = functions + holdout[str(t_idx)]
                tools = to_tools(functions)
                turn_msgs = [{"role": "user", "content": MT.DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC}]
            messages.extend(turn_msgs)
            steps, count = [], 0
            while True:
                r = ctx.chat(messages, tools=tools, max_tokens=cfg.max_tokens)
                if r.get("error"):
                    api_err = r["error"]
                    break
                messages.append(_assistant_msg(r, cfg))
                raw_markup += 1 if _raw_markup(r) else 0
                if not r["tool_calls"]:
                    break
                try:
                    calls = MT.convert_to_function_call(_decode(r["tool_calls"]))
                except Exception:
                    break  # upstream: decode failure ends the turn
                if MT.is_empty_execute_response(calls):
                    break
                results = ex.execute(calls)
                for i, (tc, res) in enumerate(zip(r["tool_calls"], results)):
                    messages.append({"role": "tool", "tool_call_id": tc.get("id") or f"call_{i}", "content": res})
                steps.append(calls)
                count += 1
                steps_total += 1
                if count > MT.MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    break
            turns.append(steps)
            if force_quit or api_err:
                break
        rec = {"turns": len(e["question"]), "steps": steps_total, "raw_tool_markup": raw_markup,
               "calls": [[c[:300] for s in t for c in s] for t in turns]}
        if api_err:
            return dict(rec, correct=False, status="api_error")
        if force_quit or len(turns) != len(gt):
            return dict(rec, correct=False, status="multi_turn:force_terminated")
        res = MT.multi_turn_checker(turns, gt, e, long_ctx)
        return dict(rec, correct=bool(res["valid"]), status="pass" if res["valid"] else res.get("error_type"),
                    checker_error=None if res["valid"] else str({k: v for k, v in res.items() if k != "valid"})[:800])

    def fn(item, ctx):
        return multi(item, ctx) if item["category"].startswith("multi_turn") else single(item, ctx)
    return fn
