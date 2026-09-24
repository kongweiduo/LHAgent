"""终端诊断的凭据遮盖边界；所有凭据均为测试用假值。"""

import json

import pytest

from lhagent.diagnostics import diagnostic


@pytest.mark.parametrize(
    "text",
    [
        "Authorization: Basic dXNlcjpwYXNz",
        "authorization=Bearer fake-token",
        'headers={"Authorization": "Basic dXNlcjpwYXNz"}',
        json.dumps({"api_key": 'prefix"secret-tail'}),
        "config={'api-key': 'prefix\\'secret-tail'}",
        "access_token=fake-token",
        "auth_token=fake-token",
        "token=fake-token",
        "password=fake-token",
        "secret=fake-token",
    ],
)
def test_credentials_are_fully_redacted(monkeypatch, text):
    """带引号、转义及鉴权方案的凭据值全部被遮盖。"""
    monkeypatch.delenv("LHAGENT_API_KEY", raising=False)
    result = diagnostic(f"request failed: {text}; retry later")
    for secret in ("dXNlcjpwYXNz", "fake-token", "prefix", "secret-tail"):
        assert secret not in result
    assert "request failed:" in result and "; retry later" in result
    assert "[redacted]" in result


def test_environment_key_and_unrelated_fields(monkeypatch):
    """当前环境密钥被遮盖，无关业务字段仍可阅读。"""
    monkeypatch.setenv("LHAGENT_API_KEY", "fake-environment-key")
    text = "fake-environment-key Bearer fake-bearer-token; input_tokens=42 tokenizer=test"
    result = diagnostic(text)
    assert "fake-environment-key" not in result and "fake-bearer-token" not in result
    assert "input_tokens=42 tokenizer=test" in result
