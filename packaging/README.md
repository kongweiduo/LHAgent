# Linux 独立运行包

运行包内含私有 CPython 和 LHAgent 依赖，不修改任务环境的 Python。
需要 Linux glibc 环境，使用与目标平台匹配的 amd64 或 arm64 包。

解压后直接运行 `./lhagent/lhagent --help`；使用
`./lhagent/lhagent --bundle-check` 执行不调用模型的自检。
可选安装：`./lhagent/install.sh /absolute/path/to/bin`，创建命令软链接，
不覆盖现有命令。安装后应保留解压目录。

## 从源码构建和验证

在仓库根目录运行：

```sh
./packaging/build.sh all packaging/dist/0.52.0
./packaging/test.sh packaging/dist/0.52.0/lhagent-0.52.0-linux-amd64.tar.gz linux/amd64
./packaging/test.sh packaging/dist/0.52.0/lhagent-0.52.0-linux-arm64.tar.gz linux/arm64
```

`build.sh` 默认构建两种架构，版本从 pyproject.toml 读取。
每个版本使用独立输出目录，避免 benchmark 的 bundle 目录混入同架构的多个包。
构建缓存和打包验证使用的 Docker 镜像不属于 benchmark 题目镜像清理范围。

## SWE-bench

在宿主源码环境运行 SWE-bench adapter，传入 `--bundle packaging/dist/0.52.0`。
宿主需要 Docker、datasets 和 swebench；运行包仅提供容器内的 Agent。
配置中的工作目录应为 `/testbed`。

每题完成做题、评分和保存结果后，删除本轮容器以及运行前不存在的题目镜像，
再执行下一题。失败同样清理；清理失败停止后续任务。运行前已有镜像保留，
不会全局清理 Docker，不提供断点续跑。补丁、日志和评分报告保存在宿主机。
