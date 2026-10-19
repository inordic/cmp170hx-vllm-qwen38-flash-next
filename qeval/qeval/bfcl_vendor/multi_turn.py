# Adapted from ShishirPatil/gorilla berkeley-function-call-leaderboard
# (bfcl_eval/eval_checker/multi_turn_eval/multi_turn_utils.py and multi_turn_checker.py),
# Apache-2.0, see LICENSE in this folder.
#
# qeval changes:
# - instances live in an explicit Executor object instead of module globals();
# - calls are evaluated against a namespace of bound methods with empty __builtins__
#   (upstream rewrites the string and calls eval() with full globals);
# - the checker logic (state check + unordered response-subset check per turn) is unchanged.
import copy
import importlib
import inspect
import json

CLASS_FILE_PATH_MAPPING = {
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI": "math_api",
    "MessageAPI": "message_api",
    "TwitterAPI": "posting_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
}
MULTI_TURN_FUNC_DOC_FILE_MAPPING = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}
STATELESS_CLASSES = ["MathAPI"]
MAXIMUM_STEP_LIMIT = 20
DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC = (
    "I have updated some more functions you can choose from. What about now?"
)
_BLOCKED = {"kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run"}
_SAFE_BUILTINS = {
    "True": True, "False": False, "None": None,
    "int": int, "float": float, "str": str, "bool": bool, "list": list, "dict": dict,
    "tuple": tuple, "len": len, "abs": abs, "min": min, "max": max, "sum": sum, "round": round,
}


class Executor:
    """One set of backend instances (one 'namespace' in upstream terms) for one test entry."""

    def __init__(self, initial_config: dict, involved_classes: list, long_context: bool = False):
        self.instances = {}
        self.namespace = {}
        for class_name in involved_classes:
            module = importlib.import_module(
                "." + CLASS_FILE_PATH_MAPPING[class_name], package=__package__ + ".func_source_code"
            )
            inst = getattr(module, class_name)()
            if class_name not in STATELESS_CLASSES:
                inst._load_scenario(copy.deepcopy(initial_config.get(class_name, {})), long_context=long_context)
            self.instances[class_name] = inst
            for method_name, method in inspect.getmembers(inst, predicate=inspect.ismethod):
                if method_name.startswith("_"):
                    continue
                self.namespace[method_name] = method

    def execute(self, func_call_list):
        results = []
        for call in func_call_list:
            try:
                fname = call.split("(")[0].strip()
                if "." in fname:
                    fname = fname.split(".")[1]
                if fname in _BLOCKED:
                    raise Exception(f"Function call {fname} is not allowed.")
                res = eval(call, {"__builtins__": _SAFE_BUILTINS}, dict(self.namespace))  # noqa: S307
                if type(res) == str:
                    pass
                elif type(res) == dict:
                    try:
                        res = json.dumps(res)
                    except Exception:
                        res = str(res)
                else:
                    res = str(res)
                results.append(res)
            except Exception as e:  # mirrors upstream: errors become the tool result
                results.append(f"Error during execution: {str(e)}")
        return results


def convert_to_function_call(function_call_list):
    """[{name: args(dict|json-str)}] -> ["name(k=repr(v),...)"] exactly like upstream FC decode."""
    if type(function_call_list) == dict:
        function_call_list = [function_call_list]
    out = []
    for function_call in function_call_list:
        for key, value in function_call.items():
            if type(value) == str:
                value = json.loads(value)
            out.append(f"{key}({','.join([f'{k}={repr(v)}' for k, v in value.items()])})")
    return out


def is_empty_execute_response(input_list: list):
    if len(input_list) == 0:
        return True
    if len(input_list) == 1 and len(input_list[0]) == 0:
        return True
    return False


def multi_turn_checker(decoded_turns, ground_truth_turns, test_entry, long_context=False):
    """decoded_turns: list[turn] of list[step] of list[call-string] (empty steps removed)."""
    initial_config = test_entry["initial_config"]
    involved_classes = test_entry["involved_classes"]
    model_ex = Executor(initial_config, involved_classes, long_context)
    gt_ex = Executor(initial_config, involved_classes, long_context)
    all_turn_model_execution_results = []
    for turn_index, gt_calls in enumerate(ground_truth_turns):
        model_steps = decoded_turns[turn_index]
        turn_model_results = []
        for step in model_steps:
            turn_model_results.extend(model_ex.execute(step))
        gt_results = gt_ex.execute(gt_calls)
        all_turn_model_execution_results.extend(turn_model_results)
        if len(gt_calls) > 0:
            if not model_steps or is_empty_execute_response(model_steps):
                return {"valid": False, "error_type": "multi_turn:empty_turn_model_response",
                        "error_message": f"Model response list is empty for turn {turn_index}"}
        if not gt_calls:
            continue
        for class_name, gt_inst in gt_ex.instances.items():
            ok, diff = _compare_instances(model_ex.instances[class_name], gt_inst)
            if not ok:
                return {"valid": False, "error_type": "multi_turn:instance_state_mismatch",
                        "error_message": f"{class_name} state differs after turn {turn_index}",
                        "differences": _short(diff)}
        ok, missing = _is_subsequence_unordered(gt_results, all_turn_model_execution_results)
        if not ok:
            return {"valid": False, "error_type": "multi_turn:execution_response_mismatch",
                    "error_message": f"missing ground-truth results for turn {turn_index}",
                    "missing_items": _short(missing)}
    return {"valid": True}


def _short(x, n=600):
    s = repr(x)
    return s if len(s) <= n else s[:n] + "..."


def _compare_instances(model_object, ground_truth_object):
    assert type(model_object) == type(ground_truth_object)
    differences = {}
    valid = True
    for attr_name in vars(ground_truth_object):
        if attr_name.startswith("_"):
            continue
        model_attr = getattr(model_object, attr_name)
        gt_attr = getattr(ground_truth_object, attr_name)
        if model_attr != gt_attr:
            valid = False
            differences[attr_name] = {"model": model_attr, "ground_truth": gt_attr}
    return valid, differences


def _is_subsequence_unordered(list1, list2):
    list2_copy = list2[:]
    missing = []
    for item in list1:
        try:
            list2_copy.remove(item)
        except ValueError:
            missing.append(item)
    return len(missing) == 0, missing
