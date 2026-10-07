# LHAgent

> LHAgent 是一个面向学术研究的轻量级 AI 智能体运行框架（harness）。

## 📝 项目简介

许多开源 Agent harness 主要面向产品应用，功能丰富，但系统也相对复杂。对于学术研究来说，理解和修改这些系统往往需要投入较多精力，不便于快速建立实验基线。

LHAgent 聚焦于 Agent harness 的基础能力，以轻量、直接的实现提供一个便于理解和修改的 baseline。研究者可以以此为起点开展方法研究和对比实验；刚接触 harness 的开发者也可以通过 LHAgent 熟悉其基本工作流程。

## ✨ 核心功能

- **轻量化 TUI**：在终端中与 Agent 快速交互，便于调试和观察运行过程。
- **Benchmark 测评**：支持 SWE-bench Lite、Verified 和 Terminal-Bench 4.0，使用各自的官方评分器。

## 🛠️ 技术栈

- **实现方式**：基于 Python 3.12 自行实现 harness，不依赖第三方 Agent 框架。
- **智能体范式**：采用 ReAct 风格的模型—工具循环，由模型决定何时调用工具，并根据工具结果继续执行。
- **内置工具**：`read`、`write`、`edit`（文件读写与修改），`ls`、`find`、`grep`（文件浏览与搜索），`bash`（执行命令）。
- **主要依赖**：`openai` 用于访问兼容 OpenAI API 的模型服务，`prompt-toolkit` 用于终端交互，`jsonschema` 用于校验工具参数；另使用 `python-dotenv` 加载环境变量、`httpx` 处理网络相关错误与重试。

## 🚀 快速开始

### 从源码运行

#### 环境要求

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

#### 安装依赖

在项目根目录运行：

```sh
uv sync
```

#### 配置模型与 API 密钥

```sh
cp lhagent.example.toml lhagent.toml
cp swebench.example.toml swebench.toml
cp .example.env .env
```

在 `lhagent.toml` 中填写实际的模型名称、上下文窗口和最大输出 token 数；示例配置中的 `tools = []` 表示禁用工具，删除这一行即可启用全部内置工具。工作目录由 `cwd` 指定，使用其他路径时需先创建对应目录。

在 `.env` 中填写模型服务的 `LHAGENT_BASE_URL` 和 `LHAGENT_API_KEY`。

#### 运行项目

```sh
uv run lhagent --config lhagent.toml
```

该命令启动交互式终端界面。也可以使用 `uv run lhagent --config lhagent.toml --instruction "Hello"` 执行单条指令。

## 📖 使用示例

评测 SWE-bench Lite 或 Verified 时，先准备配置 `swebench.toml`：填写实际模型参数，将 `[coding_agent]` 中的 `cwd` 设为 `/testbed`，并按需启用内置工具。评测还需要可用的 Docker、宿主环境中的 `LHAGENT_BASE_URL` 和 `LHAGENT_API_KEY` 环境变量。

在项目根目录运行以下命令，使用已提供的运行包从 Verified 随机抽取 3 道任务进行测评：

```sh
set -a; source .env; set +a
uv run --with datasets --with swebench python -m lhagent.evals.benchmarks.swebench.adapter \
  --bundle packaging/dist \
  --config swebench.toml \
  --variant verified \
  --count 3 --seed 42
```

评测按题串行执行：做题 → 官方评分 → 保存结果 → 清理本轮容器和新增题目镜像。
每道题只运行一次；任务级失败不会重新创建容器重跑。模型连接或流式传输失败由客户端在同一上下文下自动重试，默认最多 3 次。
所有题目结束后统一生成 summary；未完成评分的题目计为 incomplete。
失败时也会清理，运行前已有镜像保留；清理失败则停止后续题目，防止磁盘继续累积。
不支持断点续跑。每次运行的最终预测、评分报告和各次尝试的日志保存在 `swebench/<run_id>/`，
每题的原始会话 JSONL 保存在 `swebench/<run_id>/.lhagent/<task_id>/<attempt>/sessions/`，
可以直接查看其中的模型消息和工具调用记录。

### Terminal-Bench

适配器使用官方 **Harbor 0.24.0**，固定运行 **Terminal-Bench 4.0** 发布包
`terminal-bench/terminal-bench@4.0.0`，不再使用初代 `terminal-bench` 框架。
适配器自动下载固定版本的官方任务包（66 题，约 452 MB，校验官方 SHA256），
缓存在 `~/.cache/lhagent/terminalbench/4.0.0/`；Harbor 使用官方预构建镜像，
将现有 LHAgent Linux 运行包放入容器后做题。
Agent 产物和日志保存到本地，再由官方独立评分环境验证，沿用任务声明的资源和产物规则。
任务环境、评分环境及辅助服务必须提供预构建镜像；缺失时记录失败，不自动构建。

将 `terminalbench.example.toml` 复制为 `terminalbench.toml`，填写模型参数。
默认不设置 `cwd`，适配器自动采用每道题官方环境的工作目录，无需逐题改配置。
显式设置时必须使用容器中的绝对路径。需要 `curl`、Docker Compose、现有 Linux 运行包及宿主环境的 API 环境变量。
任务镜像还需 GNU `timeout`，用于在评分前终止超时的 Agent。
4.0 包含 GPU 和多容器任务，不会降低官方任务的 CPU/内存/GPU 配置。开跑前读取 Docker 主机的
CPU、内存和 GPU，任务环境或评分环境的需求超出主机时直接跳过（不拉取镜像）；预检遗漏、
在 `compose up` 阶段因资源不足失败的任务同样记为跳过，测评继续执行下一题。
镜像与运行包平台自动匹配，优先本机架构。
若任务需要 Harbor 本地构建的网络隔离辅助镜像，也会记录失败，以遵守只使用预构建环境的要求。

官方测评包的依赖与本项目的 OpenAI SDK 版本不同，因此使用独立临时 Python 环境运行
测评 Harness，容器中的 Agent 则使用 `packaging/dist` 里的运行包：

```sh
cp terminalbench.example.toml terminalbench.toml
set -a; source .env; set +a
PYTHONPATH=src LITELLM_LOCAL_MODEL_COST_MAP=True \
uv run --no-project --python 3.12 --with harbor==0.24.0 \
  python -m lhagent.evals.benchmarks.terminalbench.adapter \
  --bundle packaging/dist --config terminalbench.toml \
  --count 3 --seed 42
```

使用 `--task-ids <id> ...`（也支持 `--instance-ids`）选择任务；`--all` 或不设置
选择参数时执行全部任务。`--dataset-path /path/to/tasks` 可运行已导出的本地 Harbor 任务包，
包中必须包含官方预构建镜像信息。`--dataset` 和 `--dataset-version` 可显式选择发布包。
默认使用每道题官方的 Agent/Verifier 时间限制（多数题 Agent 预算为 8 小时）。
`--timeout` 和 `--test-timeout` 可统一覆盖为固定秒数，此时 `run_metadata.json` 中
`timeout_profile.mode` 记为 `custom`，结果不能直接当作官方预算成绩。
`--official-timeouts` 已是默认行为，仅为兼容保留。

每题串行运行一次，不使用 Harbor 的任务重试：下载本题镜像 → 放入 LHAgent → 做题 →
保存轨迹和产物 → 官方评分 → 保存结果并清理 → 下一题。
清理仅针对本题 Compose 服务、卷及新增镜像，保留运行前已有镜像；清理失败停止后续任务。
不支持断点续跑，已有 `run_id` 不可覆盖。输出保存到 `terminalbench/<run_id>/`：

- `results.json`：汇总及原始 Harbor 评分结果。
- `logs/run_evaluation/<run_id>/results.json`：汇总报告，与 SWE 官方报告目录层级对应。
- `logs/run_evaluation/<run_id>/lhagent/<task_id>/report.json`：单题原始 Harbor 报告。
- `summary.json`：resolved、unresolved、incomplete、skipped 汇总。`accuracy` 分母包含全部选中任务，
  `accuracy_on_runnable` 分母扣除因主机资源不足而跳过的任务；`skipped_reasons` 记录跳过原因。
- `predictions.jsonl`：成功完成评分的任务结果（不是 SWE 的补丁）。
- `logs/attempts/<task_id>/1/`：stdout、stderr、错误日志及 `harbor/` 原始日志、锁文件和产物。
- `logs/<run_id>.failures.json`：运行或评分失败记录。
- `.lhagent/<task_id>/1/sessions/`：原始会话 JSONL，与 SWE 路径层级一致，失败时也尝试保存。

测试未通过计为 unresolved；运行、超时或评分异常计为 incomplete，并返回非零退出码。
因主机资源不足跳过的任务计为 skipped，不影响退出码。

## 🎯 项目亮点

- **轻量易改**：聚焦基础 harness 能力，便于作为研究 baseline 进行修改和扩展。
- **交互与评测兼顾**：同一 Agent 可用于终端快速调试，也可接入 Benchmark 测评。
- **过程可追踪**：通过配置文件控制运行参数，并保存会话记录，方便回看实验过程。

## 📊 性能评估

## 🔮 未来计划

- [ ] 完善终端交互和整体使用体验。
- [ ] 持续开发并完善 harness 的核心功能。
- [ ] 接入更多 Benchmark，开展更广泛的测评。

## 🤝 贡献指南

欢迎提出Issue和Pull Request！

## 📄 许可证

本项目采用 [MIT License](LICENSE)。

## 👤 作者

- GitHub: [@kongweiduo](https://github.com/kongweiduo)
- Email: victorkong355@gmail.com

## 🙏 致谢
