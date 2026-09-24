"""工具登记、schema 和参数校验；无效调用不得进入处理器。"""

import math

import pytest

from lhagent.harness.tools.registry import ToolRegistry, describe_tools
from lhagent.harness.tools.validation import (
    prepare_tool_call,
    validate_arguments,
    validate_schema,
)


async def handler(arguments, context):
    """校验测试的哨兵；一旦错误地执行处理器就使测试失败。"""
    raise AssertionError("Validation must not execute handlers")


def make_tool(name="read", parameters=None):
    """构造可覆盖 schema 的工具定义，绑定禁止执行的哨兵。"""
    return {
        "name": name,
        "description": "Read files",
        "parameters": parameters
        if parameters is not None
        else {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "count": {"type": "integer", "default": 2},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "handler": handler,
    }


def call(arguments, name="read"):
    """构造名称及参数可变的校验输入。"""
    return {"call_id": "call-1", "name": name, "arguments": arguments}


@pytest.mark.parametrize(
    "schema",
    [
        {},
        {"type": "array"},
        {"type": "object", "$schema": "http://json-schema.org/draft-07/schema#"},
        {"type": "object", "properties": {"x": {"$ref": "#/$defs/x"}}},
        {"type": "object", "allOf": [{"properties": {"x": {"$dynamicRef": "#x"}}}]},
        {"type": "object", "$defs": {"x": {"unknownKeyword": 1}}},
        {"type": "object", "properties": {"x": {"type": "not-a-type"}}},
        {"type": "object", "propertyNames": {"extra": True}},
        {
            "type": "object",
            "properties": {"x": {"type": "string", "format": "date", "$schema": "draft-07"}},
        },
    ],
)
def test_invalid_schema_rejected_at_registration(schema):
    """非法 schema 在登记工具时拒绝。"""
    registry = ToolRegistry()
    with pytest.raises(ValueError):
        registry.register(make_tool(parameters=schema))
    assert registry.list_tools() == []


def test_property_names_are_not_schema_keywords():
    """属性名称不能被误判为 schema 关键字。"""
    schema = {
        "type": "object",
        "properties": {"customKeyword": {"type": "string"}},
        "patternProperties": {"^x": {"type": "number"}},
        "$defs": {"anything": {"type": "null"}},
        "dependentSchemas": {"customKeyword": {"required": ["x1"]}},
    }
    validate_schema(schema)


def test_registry_selection_and_description():
    """目录选择与模型可见描述使用同一工具定义。"""
    registry = ToolRegistry()
    read = make_tool()
    write = make_tool("write")
    registry.register(read)
    registry.register(write)
    assert registry.get("read") is read
    assert registry.list_tools() == [read, write]
    assert registry.select(["write", "read", "write"]) == [write, read]
    with pytest.raises(ValueError, match="already registered"):
        registry.register(read)
    with pytest.raises(KeyError, match="missing"):
        registry.select(["read", "missing"])
    with pytest.raises(KeyError, match="missing"):
        registry.get("missing")
    descriptions = describe_tools(registry.select(["write"]))
    assert descriptions == [
        {"name": "write", "description": "Read files", "parameters": write["parameters"]}
    ]
    descriptions[0]["parameters"]["required"].clear()
    assert write["parameters"]["required"] == ["path"]


@pytest.mark.parametrize(
    "arguments,location",
    [
        ({}, "$"),
        ({"path": 42}, '$["path"]'),
        ({"path": "file", "extra": True}, "$"),
        ({"path": "file", "count": "2"}, '$["count"]'),
        ({"path": "file", "count": math.nan}, '$["count"]'),
        ({"path": "file", "count": (1, 2)}, '$["count"]'),
        ({1: "file"}, "$"),
        ('{"path": "file"}', "$"),
        (None, "$"),
    ],
)
def test_invalid_arguments_do_not_execute(arguments, location):
    """非法模型参数不得进入处理器。"""
    result = prepare_tool_call(call(arguments), [make_tool()])
    assert result["status"] == "validation_error"
    assert result["call_id"] == "call-1"
    assert location in result["error"]
    assert result["output"] is None


def test_nested_error_path_and_circular_input():
    """嵌套错误路径准确，循环输入被安全拒绝。"""
    tool = make_tool(
        parameters={
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                    },
                }
            },
        }
    )
    with pytest.raises(ValueError, match=r'\$\["items"\]\[0\]\["value"\]'):
        validate_arguments(tool, {"items": [{"value": "wrong"}]})
    circular = {}
    circular["self"] = circular
    with pytest.raises(ValueError, match="circular"):
        validate_arguments(tool, circular)


def test_enabled_tools_and_valid_arguments():
    """仅启用工具和有效参数可形成待执行调用。"""
    tool = make_tool()
    arguments = {"path": "file"}
    prepared = prepare_tool_call(call(arguments), [tool])
    assert prepared == {"call": call(arguments), "tool": tool, "arguments": arguments}
    assert prepared["arguments"] is arguments
    assert "count" not in arguments
    unknown = prepare_tool_call(call(arguments, "other"), [tool])
    assert unknown["status"] == "validation_error"
    assert unknown["call_id"] == "call-1"
    assert "other" in unknown["error"]


def test_format_is_only_an_annotation():
    """format 仅作注解，不增加未声明的格式校验。"""
    tool = make_tool(
        parameters={
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "format": "date",
                }
            },
        }
    )
    assert validate_arguments(tool, {"date": "not a date"}) == {"date": "not a date"}
