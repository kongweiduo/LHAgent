# 0.52.0 构建验证（2026-09-24）

- 全量测试：`uv run --offline pytest -q --tb=short`，943 passed。
- 此后新增两项 Agent 失败/超时容器清理测试；SWE-bench 专项最终为 27 passed。
- Ruff check、format check、`uv lock --check --offline`、git diff check 均通过。
- amd64、arm64 独立包构建成功，版本为 0.52.0，私有 CPython 3.12.14。
- 两种架构均通过 `test.sh` 的 no-python、task-python 容器验证。
- 已核对两个压缩包中的 adapter.py、grader.py 与工作区一致，SHA-256 校验通过。

宿主为 macOS arm64 / Docker Desktop，amd64 使用仿真执行。容器测试覆盖非 root、
只读根文件系统、禁网、自检、安装、路径迁移及任务 Python 环境隔离。
SWE-bench 调度测试覆盖逐题评分后再清理、失败清理、已有镜像保护、清理失败停止，
以及从已保存的官方逐题报告汇总；没有调用真实模型或运行完整 SWE-bench。

产物位于 `packaging/dist/0.52.0/`（不纳入 Git），SHA-256：

```text
8056dbcddb9323cf041c725bbfcd494d707cc1a7cb2ffd5be9f8066fc1df477b  lhagent-0.52.0-linux-amd64.tar.gz
84d2ccf5df118d43f6a8feb3fc16c5873fd85a8e8cdcb2cffc27462d626d13bb  lhagent-0.52.0-linux-arm64.tar.gz
```

复验命令见本目录 README.md。以下保留 0.51.0 的历史验证记录。

# 2026-09-24 构建验证

产物：LHAgent 0.51.0，私有 CPython 3.12.14，依赖取自 uv.lock。
构建宿主：macOS arm64 / Docker Desktop。Linux amd64 测试通过 Docker 的跨架构执行完成，
并非在原生 amd64 宿主上测试。

| 验证 | 结果 |
| --- | --- |
| 项目全量 pytest | 918 passed |
| Ruff check / format | 通过 |
| 最终进程解析调整后的 bash 测试 | 18 passed |
| 安装包私有解释器下的 Linux find/grep/bash 测试 | 43 passed |
| Linux arm64，Debian bookworm，无系统 Python、无 ps | 通过 |
| Linux arm64，Debian bookworm，任务 Python 3.11.2、无 ps | 通过 |
| Linux amd64，Debian bookworm，无系统 Python、无 ps | 通过 |
| Linux amd64，Debian bookworm，任务 Python 3.11.2、无 ps | 通过 |
| 现有 task_1_rate_limiter-runner:latest，arm64，任务 Python 3.11.8、无 ps | 通过 |

容器检查均在禁网、只读根文件系统、非 root 用户下进行；包解压至临时可执行挂载，
更改安装路径（含空格），再去掉包目录写权限。覆盖 CLI 启动、会话列表、真实搜索
worker、bash、后台子进程终止、命令链接、重复安装拒绝、环境变量继承，以及
PYTHONHOME/PYTHONPATH 指向不存在的任务路径时 agent 的运行。

现有 benchmark 镜像验证只启动临时容器，没有向原镜像安装或更新软件。
任务 Python 仍为 `/usr/local/bin/python`；agent Python 位于解压包的
`runtime/bin/python3.12`。没有调用真实模型 API；模型连接和具体 benchmark
成绩不属于本次离线打包验证。

已核对压缩包中的 launcher、安装器、自检脚本、说明和修改过的工具源码与工作区
一致；没有打包个人 `.env`、`.venv`、会话或个人配置。SHA-256：

```text
fca9131241512ae5136fa861fad0da459610cff36ef8bb9f9d0a3ccbb66214cf  lhagent-0.51.0-linux-amd64.tar.gz
6e78da938d08ba8476886b3113af05d036f83debb4d1dfb17650f4eac5708485  lhagent-0.51.0-linux-arm64.tar.gz
```

复验命令见 `README.md`，构建脚本为 `build.sh`，容器测试入口为 `test.sh`。
