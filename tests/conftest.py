"""隔离离线测试的工作目录、个人配置与凭据。"""

import os

import pytest


@pytest.fixture(autouse=True)
def isolated_configuration(tmp_path, monkeypatch):
    """在临时目录运行测试，并清除宿主的客户端环境变量。"""
    monkeypatch.chdir(tmp_path)
    for name in tuple(os.environ):
        if name.startswith("LHAGENT_"):
            monkeypatch.delenv(name)
