# LHAgent

> LHAgent 是一个面向学术研究的轻量级 AI 智能体运行框架（harness）。

## 📝 项目简介

许多开源 Agent harness 主要面向产品应用，功能丰富，但系统也相对复杂。对于学术研究来说，理解和修改这些系统往往需要投入较多精力，不便于快速建立实验基线。

LHAgent 聚焦于 Agent harness 的基础能力，以轻量、直接的实现提供一个便于理解和修改的 baseline。研究者可以以此为起点开展方法研究和对比实验；刚接触 harness 的开发者也可以通过 LHAgent 熟悉其基本工作流程。

## ✨ 核心功能

- **轻量化 TUI**：在终端中与 Agent 快速交互，便于调试和观察运行过程。
- **Benchmark 测评**：通过评测适配器运行基准任务并评估结果；目前提供 SWE-bench Lite 适配器，可按研究需要扩展其他 Benchmark。

## 🛠️ 技术栈

- **实现方式**：基于 Python 3.12 自行实现 harness，不依赖第三方 Agent 框架。
- **智能体范式**：采用 ReAct 风格的模型—工具循环，由模型决定何时调用工具，并根据工具结果继续执行。
- **内置工具**：`read`、`write`、`edit`（文件读写与修改），`ls`、`find`、`grep`（文件浏览与搜索），`bash`（执行命令）。
- **主要依赖**：`openai` 用于访问兼容 OpenAI API 的模型服务，`prompt-toolkit` 用于终端交互，`jsonschema` 用于校验工具参数；另使用 `python-dotenv` 加载环境变量、`httpx` 处理网络相关错误与重试。

## 🚀 快速开始

### 环境要求

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

### 安装依赖

在项目根目录运行：

```sh
uv sync
```

### 配置模型与 API 密钥

```sh
cp lhagent.example.toml lhagent.toml
cp .example.env .env
mkdir -p manual-workspace
```

在 `lhagent.toml` 中填写实际的模型名称、上下文窗口和最大输出 token 数；示例配置中的 `tools = []` 表示禁用工具，删除这一行即可启用全部内置工具。工作目录由 `cwd` 指定，使用其他路径时需先创建对应目录。

在 `.env` 中填写模型服务的 `LHAGENT_BASE_URL` 和 `LHAGENT_API_KEY`。

### 运行项目

```sh
uv run lhagent --config lhagent.toml
```

该命令启动交互式终端界面。也可以使用 `uv run lhagent --config lhagent.toml --instruction "Hello"` 执行单条指令。

## 📖 使用示例

以 SWE-bench Lite 为例，先准备评测配置 `swebench.toml`：填写实际模型参数，将 `[coding_agent]` 中的 `cwd` 设为 `/testbed`，并按需启用内置工具。评测还需要可用的 Docker、宿主环境中的 `LHAGENT_BASE_URL` 和 `LHAGENT_API_KEY` 环境变量。

在项目根目录运行以下命令，构建运行包并随机抽取 3 道任务进行测评：

```sh
./packaging/build.sh
uv run --with datasets --with swebench python -m lhagent.evals.benchmarks.swebench.adapter \
  --bundle packaging/dist \
  --config swebench.toml \
  --count 3 --seed 42
```

评测按题串行执行：做题 → 官方评分 → 保存结果 → 清理本轮容器和新增题目镜像。
失败时也会清理，运行前已有镜像保留；清理失败则停止后续题目，防止磁盘继续累积。
不支持断点续跑。详细行为见 [SWE-bench 说明](src/lhagent/evals/benchmarks/swebench/README.md)。

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
