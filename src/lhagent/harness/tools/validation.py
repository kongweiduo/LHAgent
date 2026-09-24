"""校验工具可用性和参数 schema，不执行工具或自动修复模型调用。"""

import json
import math

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .types import PreparedToolCall, ToolCall, ToolDefinition, ToolResult

_DIALECT = Draft202012Validator.META_SCHEMA["$id"]
_KEYWORDS = frozenset(
    """$schema $id $anchor $dynamicAnchor $vocabulary $defs $comment
    type enum const multipleOf maximum exclusiveMaximum minimum exclusiveMinimum
    maxLength minLength pattern items prefixItems contains minContains maxContains
    maxItems minItems uniqueItems maxProperties minProperties required properties
    patternProperties additionalProperties propertyNames dependentRequired
    dependentSchemas unevaluatedProperties unevaluatedItems allOf anyOf oneOf not
    if then else title description default deprecated readOnly writeOnly examples
    format contentEncoding contentMediaType contentSchema""".split()
)
_MAPPED_SCHEMAS = {"$defs", "properties", "patternProperties", "dependentSchemas"}
_SINGLE_SCHEMAS = {
    "additionalProperties",
    "propertyNames",
    "items",
    "contains",
    "unevaluatedProperties",
    "unevaluatedItems",
    "not",
    "if",
    "then",
    "else",
    "contentSchema",
}
_LIST_SCHEMAS = {"prefixItems", "allOf", "anyOf", "oneOf"}


def _location(parts: tuple[object, ...]) -> str:
    """将参数路径格式化为错误位置，不输出参数值。"""
    return "$" + "".join(
        f"[{json.dumps(part)}]" if isinstance(part, str) else f"[{part}]" for part in parts
    )


def _check_keywords(schema: object, path: tuple[object, ...] = ()) -> None:
    """递归检查受支持的 schema 关键字，属性名不当作关键字处理。"""
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        return  # 畸形子 schema 由 check_schema 报错。
    for key, value in schema.items():
        location = path + (key,)
        if key in {"$ref", "$dynamicRef"} or key not in _KEYWORDS:
            raise ValueError(f"Unsupported schema keyword at {_location(location)}: {key}")
        if key == "$schema" and value != _DIALECT:
            raise ValueError(f"Unsupported schema dialect at {_location(location)}: {value}")
        if key in _MAPPED_SCHEMAS and isinstance(value, dict):
            for name, child in value.items():
                _check_keywords(child, location + (name,))
        elif key in _SINGLE_SCHEMAS:
            _check_keywords(value, location)
        elif key in _LIST_SCHEMAS and isinstance(value, list):
            for index, child in enumerate(value):
                _check_keywords(child, location + (index,))


def _check_json(value: object, path: tuple[object, ...] = (), seen: set[int] | None = None) -> None:
    """校验 JSON 值并检测循环引用，不接受非有限数字。"""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if seen is None:
        seen = set()
    if isinstance(value, (dict, list)):
        if id(value) in seen:
            raise ValueError(f"{_location(path)}: circular JSON value")
        seen.add(id(value))
        try:
            if isinstance(value, dict):
                for key, child in value.items():
                    if not isinstance(key, str):
                        raise ValueError(f"{_location(path)}: JSON object keys must be strings")
                    _check_json(child, path + (key,), seen)
            else:
                for index, child in enumerate(value):
                    _check_json(child, path + (index,), seen)
        finally:
            seen.remove(id(value))
        return
    raise ValueError(f"{_location(path)}: not a JSON value")


def validate_schema(parameters: dict[str, object]) -> None:
    """登记时检查 Draft 2020-12 schema，拒绝引用和未知关键字。"""
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise ValueError("Tool schema root must declare type: object")
    _check_json(parameters)
    _check_keywords(parameters)
    try:
        Draft202012Validator.check_schema(parameters)
    except SchemaError as exc:
        raise ValueError(
            f"Invalid tool schema at {_location(tuple(exc.absolute_path))}: {exc.message}"
        ) from exc


def validate_arguments(tool: ToolDefinition, arguments: object) -> dict[str, object]:
    """检查 JSON 对象参数，不转换类型、不应用默认值。"""
    if not isinstance(arguments, dict):
        raise ValueError("$: tool arguments must be a JSON object")
    _check_json(arguments)
    error = next(Draft202012Validator(tool["parameters"]).iter_errors(arguments), None)
    if error is not None:
        raise ValueError(f"{_location(tuple(error.absolute_path))}: {error.message}")
    return arguments


def prepare_tool_call(
    call: ToolCall, active_tools: list[ToolDefinition]
) -> PreparedToolCall | ToolResult:
    """仅解析当前可用工具；校验失败生成保留调用 ID 的结果。"""
    tool = next((item for item in active_tools if item["name"] == call["name"]), None)
    if tool is None:
        return {
            "call_id": call["call_id"],
            "name": call["name"],
            "status": "validation_error",
            "output": None,
            "error": f"Tool is not enabled: {call['name']}",
        }
    try:
        arguments = validate_arguments(tool, call["arguments"])
    except ValueError as exc:
        return {
            "call_id": call["call_id"],
            "name": call["name"],
            "status": "validation_error",
            "output": None,
            "error": str(exc),
        }
    return {"call": call, "tool": tool, "arguments": arguments}
