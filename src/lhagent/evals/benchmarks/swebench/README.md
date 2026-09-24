# SWE-bench Lite

先构建两种 Linux 运行包（内含独立 Python，不是题目环境镜像）：

```bash
./packaging/build.sh linux/amd64
./packaging/build.sh linux/arm64
```

只需传入产物目录：

```bash
python -m lhagent.evals.benchmarks.swebench.adapter \
  --bundle packaging/dist \
  --config src/lhagent/evals/benchmarks/swebench/lhagent.toml \
  --count 3 --seed 42
```

也兼容 `--bundle path/to/bundle.tar.gz`。目录中每个平台只能有一个 `.tar.gz`
运行包；根据包内 `lhagent/TARGET` 识别架构，不依赖文件名。

适配器查询每道题环境镜像的 manifest（查询失败时尝试本地镜像），与可用运行包
取交集。优先 Docker daemon 的原生架构，其次 amd64、arm64。创建做题容器和
官方评分容器时都显式指定所选平台。跨架构运行需要 Docker daemon 已启用仿真；
Docker Desktop 通常支持，生产批量评测建议使用对应架构的 Linux 主机。
不会自动构建或替换题目环境，也不会更改镜像名来猜测其他架构是否存在。

`logs/lhagent-swebench/<run-id>.platforms.json` 记录镜像与平台的对应关系。
失败任务记录在 `<run-id>.failures.json` 和逐题 `.error.log`，不作为空补丁提交。
全部失败时跳过评分；有任意执行失败时适配器返回非零退出码。
`--max-workers` 控制评分并发；当前做题阶段依次运行。

宿主 Python 环境需安装 `datasets`、`swebench`（含 Docker SDK）；模型设置见配置文件。
