"""LHAgent 包入口；导入不启动运行，也不读取凭据。"""


def main() -> None:
    """控制台入口；按需导入 CLI，避免包导入触发终端初始化。"""
    from .cli import main as run_cli

    run_cli()
