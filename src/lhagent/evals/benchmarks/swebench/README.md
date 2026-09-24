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
每道题依次完成做题、官方评分、结果保存和 Docker 清理后，才开始下一题。
补丁逐题写入 predictions.jsonl 并刷新；官方评分日志和逐题报告保存在
`logs/evaluation/<run-id>/`，最后使用官方报告函数汇总，汇总不会重新下载镜像。
做题失败也会清理；评分异常时会按本轮唯一标签清理遗留容器。
仅删除本轮使用且运行前不存在的题目镜像，不使用全局 prune，也不强制删除镜像。
运行前已有的镜像保留；共享层是否释放由 Docker 决定。清理失败时停止后续题目，
避免磁盘继续累积；执行、评分或清理失败均返回非零退出码。不提供断点续跑。
`--max-workers` 保留兼容，但每次只提交一道题评分，整个流程串行。
该策略限制题目镜像的累积，仍需为单题镜像和运行时文件预留足够空间。

宿主 Python 环境需安装 `datasets`、`swebench`（含 Docker SDK）；模型设置见配置文件。
