"""封装一次 OpenAI 兼容网络尝试及协议响应解析。

不决定重试策略、不统计跨尝试耗时、不修改业务消息，也不执行工具调用。
底层失败交给 Client 统一处理；避免底层 SDK 隐式重试与 Client 重试叠加。
异常须保留供 Client 分类的状态、错误码和说明，不提前压成只剩展示文本的
异常；Client 在分类后脱敏，不将原始响应体或凭据交给 loop。
"""

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from typing import Literal, TypedDict

from openai import APIError, AsyncOpenAI, AsyncStream

from .config import ClientConfig, validate_config
from .errors import ProtocolError
from .types import ClientRequest, TokenUsage

# 固定协议约束，不属于用户可调生成参数。
# 参数只进入 JSON body；拒绝 SDK options 及协议控制字段，包括 extra_body 后门。
_RESERVED_PARAMETERS = {
    "api_key",
    "base_url",
    "url",
    "headers",
    "default_headers",
    "extra_headers",
    "extra_query",
    "extra_body",
    "query",
    "body",
    "options",
    "timeout",
    "http_client",
    "max_retries",
    "organization",
    "project",
    "webhook_secret",
    "stream",
    "stream_options",
    "n",
    "model",
    "messages",
    "call_id",
    "max_tokens",
    "max_completion_tokens",
}


class ResponseDelta(TypedDict):
    """单个协议内容增量；工具索引属于当前响应的 tool_calls 索引空间。

    文本及推理的 tool_index 为 None，工具增量须带协议提供的工具索引。
    Client 按类型和工具索引维护状态，分配公共 StreamEvent 的 block_index；
    工具索引不能直接当作混合文本、推理和工具内容列表的下标。
    文本和推理 data 为 {"text": str}；工具 data 按分片实际提供的字段包含
    call_id、name、arguments_json，不补齐缺失项，也不解析残缺 JSON。
    """

    type: Literal["text", "reasoning", "tool_call"]
    tool_index: int | None
    data: dict[str, object]


class ResponseChunk(TypedDict):
    """一个协议分片解析后的全部增量及分片级元数据。

    deltas 可为空，也可同时包含文本、推理和多个工具增量，不只取第一项。
    usage 和 finish_reason 属于整个分片，不能为每条增量重复更新统计或终结。
    usage 缺失为 None；finish_reason 保留协议值，由 Client 映射公共状态。
    该结构不代表流已经完整读完。
    """

    deltas: list[ResponseDelta]
    usage: TokenUsage | None
    finish_reason: str | None


class Transport:
    """持有底层通信资源，提供单次尝试的接口，不保存会话状态。"""

    def __init__(self, config: ClientConfig) -> None:
        """接收连接及鉴权配置，不发送请求，也不读取业务上下文。"""
        validate_config(config)
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=0,
        )
        self._streams: set[_ResponseStream] = set()
        self._closed = False
        self._close_lock = asyncio.Lock()

    async def open_stream(self, request: ClientRequest) -> AsyncIterator[ResponseChunk]:
        """建立一次请求并返回解析后的响应流，不进行自动重试。

        await 阶段完成请求建立；返回后的迭代阶段仅负责读取和解析，从而让上层
        区分可考虑重试的建立失败与不可自动重放的流读取失败。
        迭代器结束、关闭或取消时释放本次响应资源；协议失败向上层报告。
        """
        if self._closed:
            raise RuntimeError("transport is closed")
        body = _request_body(request)
        raw = await self._client.post(
            "/chat/completions",
            body=body,
            cast_to=object,
            stream=True,
            stream_cls=AsyncStream[object],
        )
        stream = _ResponseStream(raw, self._streams)
        self._streams.add(stream)
        if self._closed:
            await stream.aclose()
            raise RuntimeError("transport is closed")
        return stream

    async def close(self) -> None:
        """释放本通信对象持有的连接资源，可重复调用，不关闭远端服务。"""
        async with self._close_lock:
            self._closed = True
            try:
                for stream in tuple(self._streams):
                    await stream.aclose()
            finally:
                await self._client.close()


class _ResponseStream(AsyncIterator[ResponseChunk]):
    """显式拥有响应，尚未开始迭代时 aclose 也会释放连接。"""

    def __init__(self, raw: AsyncStream, streams: set) -> None:
        """持有一次 SDK 响应流及所属活动流集合，记录幂等关闭状态。"""
        self._raw = raw
        self._streams = streams
        self._closed = False

    async def __anext__(self) -> ResponseChunk:
        """读取并解析分片；结束或失败时关闭响应资源。"""
        if self._closed:
            raise StopAsyncIteration
        try:
            raw = await anext(self._raw)
            return decode_chunk(raw)
        except APIError as error:
            # SDK 流内错误保留 body/code，但默认没有响应头和状态。
            error.response = self._raw.response
            await self.aclose()
            raise
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            await self.aclose()
            raise ProtocolError("stream data must be UTF-8 JSON") from error
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        """关闭底层响应流，提前停止消费时也必须调用。"""
        if not self._closed:
            await self._raw.close()
            self._closed = True
            self._streams.discard(self)


def _request_body(request: ClientRequest) -> dict[str, object]:
    """组装单次协议请求；保留字段不得被生成参数覆盖。"""
    parameters = request["parameters"]
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping")
    if any(not isinstance(key, str) or key in _RESERVED_PARAMETERS for key in parameters):
        raise ValueError("parameters contains reserved fields")
    body = dict(parameters)
    if "tools" in body:
        tools = body["tools"]
        if not isinstance(tools, list):
            raise TypeError("tools must be a list")
        # 内部工具描述不绑定协议，在发送边界补上兼容接口要求的包装。
        body["tools"] = [_tool_body(tool) for tool in tools]
    if "max_output_tokens" in body:
        limit = body.pop("max_output_tokens")
        if type(limit) is not int or limit <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        body["max_completion_tokens"] = limit
    if not isinstance(request["model"], str) or not request["model"]:
        raise ValueError("model must be a nonempty string")
    body.update(
        model=request["model"],
        messages=[_message_body(m) for m in request["messages"]],
        stream=True,
        n=1,
        stream_options={"include_usage": True},
    )
    return body


def _tool_body(tool: object) -> dict[str, object]:
    """将内部工具描述包装为协议要求的函数定义。"""
    if not isinstance(tool, Mapping):
        raise TypeError("tool must be an object")
    if not isinstance(tool.get("name"), str) or not tool["name"]:
        raise ValueError("tool function requires a name")
    if not isinstance(tool.get("parameters"), Mapping):
        raise ValueError("tool function requires parameters")
    return {"type": "function", "function": dict(tool)}


def _json_text(value: object) -> str:
    """序列化严格 JSON；编码失败转为本地参数错误。"""
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError):
        raise ValueError("message content must be JSON serializable") from None


def _message_body(message: Mapping[str, object]) -> dict[str, object]:
    """转换消息及内容块到兼容协议，拒绝角色不支持的内容形状。"""
    role = message["role"]
    if role not in ("system", "user", "assistant", "tool"):
        raise ValueError("unsupported message role")
    texts, reasoning, tools = [], [], []
    for block in message["content"]:
        kind = block["type"]
        if kind == "text" or (kind == "reasoning" and role == "assistant"):
            if not isinstance(block["text"], str):
                raise TypeError("content text must be a string")
            (texts if kind == "text" else reasoning).append(block["text"])
        elif kind == "tool_call" and role == "assistant":
            if block.get("complete") is not True:
                raise ValueError("tool call must be complete before replay")
            if any(not isinstance(block.get(k), str) or not block[k] for k in ("call_id", "name")):
                raise ValueError("tool call requires call_id and name")
            arguments = block.get("arguments_json")
            if arguments is None:
                if not isinstance(block.get("arguments"), dict):
                    raise ValueError("tool call requires arguments")
                arguments = _json_text(block["arguments"])
            try:
                if not isinstance(arguments, str) or not isinstance(json.loads(arguments), dict):
                    raise ValueError
            except ValueError:
                raise ValueError("tool arguments must encode a JSON object") from None
            tools.append(
                {
                    "id": block["call_id"],
                    "type": "function",
                    "function": {"name": block["name"], "arguments": arguments},
                }
            )
        elif kind == "tool_result" and role == "tool":
            value = block["content"]
            texts.append(value if isinstance(value, str) else _json_text(value))
        else:
            raise ValueError("content block is unsupported for message role")
    result = {"role": role, "content": "".join(texts)}
    if reasoning:
        result["reasoning_content"] = "".join(reasoning)
    if tools:
        result["tool_calls"] = tools
    if role == "tool":
        call_id = message.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("tool message requires tool_call_id")
        result["tool_call_id"] = call_id
    if "name" in message:
        result["name"] = message["name"]
    return result


def decode_chunk(raw_chunk: Mapping[str, object]) -> ResponseChunk:
    """把 OpenAI 兼容分片转换为内部响应分片，不累积整条消息。

    只解析协议字段，包括全部内容增量、用量及结束原因；不累积内容或分配公共
    block_index。保留每条工具增量的协议索引，缺少必要关联信息时报协议错误。
    """
    raw = _mapping(raw_chunk, "chunk")
    if "error" in raw:
        raise ProtocolError("chunk.error is not a completion chunk")
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) > 1:
        raise ProtocolError("choices must be a list with at most one choice")

    deltas: list[ResponseDelta] = []
    finish_reason = None
    if choices:
        choice = _mapping(choices[0], "choices[]")
        if _index(choice.get("index"), "choices[].index") != 0:
            raise ProtocolError("only choice index 0 is supported")
        delta = _mapping(choice.get("delta"), "choices[].delta")
        finish_reason = _string(choice.get("finish_reason"), "finish_reason")
        if finish_reason == "":
            raise ProtocolError("finish_reason must be nonempty")
        role = delta.get("role")
        if role is not None and role != "assistant":
            raise ProtocolError("delta.role must be assistant")
        for field in ("function_call", "refusal", "audio"):
            if delta.get(field) is not None:
                raise ProtocolError(f"delta.{field} is unsupported")
        for field, kind in (("content", "text"), ("reasoning_content", "reasoning")):
            text = _string(delta.get(field), f"delta.{field}")
            if text is not None:
                deltas.append({"type": kind, "tool_index": None, "data": {"text": text}})

        tools = delta.get("tool_calls")
        if tools is not None:
            if not isinstance(tools, list):
                raise ProtocolError("delta.tool_calls must be a list")
            for raw_tool in tools:
                tool = _mapping(raw_tool, "tool_calls[]")
                index = _index(tool.get("index"), "tool_calls[].index")
                if tool.get("type") not in (None, "function"):
                    raise ProtocolError("tool_calls[].type must be function")
                data: dict[str, object] = {}
                call_id = _string(tool.get("id"), "tool_calls[].id")
                if call_id is not None:
                    if not call_id:
                        raise ProtocolError("tool_calls[].id must be nonempty")
                    data["call_id"] = call_id
                function = tool.get("function")
                if function is not None:
                    function = _mapping(function, "tool_calls[].function")
                    name = _string(function.get("name"), "function.name")
                    arguments = _string(function.get("arguments"), "function.arguments")
                    if name is not None:
                        if not name:
                            raise ProtocolError("function.name must be nonempty")
                        data["name"] = name
                    if arguments is not None:
                        data["arguments_json"] = arguments
                deltas.append({"type": "tool_call", "tool_index": index, "data": data})

    return {
        "deltas": deltas,
        "usage": _decode_usage(raw.get("usage")),
        "finish_reason": finish_reason,
    }


def _mapping(value: object, field: str) -> Mapping[str, object]:
    """要求协议字段为映射，否则抛出 ProtocolError。"""
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{field} must be an object")
    return value


def _string(value: object, field: str) -> str | None:
    """要求协议字段为字符串或空值，不做隐式转换。"""
    if value is not None and not isinstance(value, str):
        raise ProtocolError(f"{field} must be a string or null")
    return value


def _index(value: object, field: str) -> int:
    """要求协议索引为非负整数，拒绝布尔值。"""
    if type(value) is not int or value < 0:
        raise ProtocolError(f"{field} must be a nonnegative integer")
    return value


def _decode_usage(value: object) -> TokenUsage | None:
    """读取已知用量计数，缺失计数保留 None，不自行估算。"""
    if value is None:
        return None
    usage = _mapping(value, "usage")

    def counter(source: Mapping[str, object], field: str) -> int | None:
        value = source.get(field)
        return None if value is None else _index(value, f"usage.{field}")

    details = usage.get("prompt_tokens_details")
    cache_read = None
    if details is not None:
        cache_read = counter(_mapping(details, "usage.prompt_tokens_details"), "cached_tokens")
    return {
        "input_tokens": counter(usage, "prompt_tokens"),
        "output_tokens": counter(usage, "completion_tokens"),
        "total_tokens": counter(usage, "total_tokens"),
        "cache_read_tokens": cache_read,
        "cache_write_tokens": None,
    }
