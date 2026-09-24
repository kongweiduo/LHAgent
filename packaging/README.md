# LHAgent 容器离线安装包

在有 Docker/buildx 的机器上构建（包括 macOS）；构建期间需要网络：

```sh
./packaging/build.sh linux/amd64
./packaging/build.sh linux/arm64
```

产物位于 `packaging/dist/lhagent-0.51.0-linux-{amd64,arm64}.tar.gz`，另有
SHA-256 校验文件。版本来自项目元数据。构建器只接收白名单中的源代码、锁文件
和打包脚本，不发送个人 `.env`、配置、会话或 `.venv` 到 Docker 构建上下文。
依赖由 `uv.lock` 固定并验证哈希，Python 固定为 3.12.14，uv 固定为 0.12.7。
这保证版本固定；不宣称每次打包的字节完全相同。

## 注入已有任务容器

选择与容器 CPU 一致的包。以下为 amd64 示例：

```sh
docker cp packaging/dist/lhagent-0.51.0-linux-amd64.tar.gz TASK:/tmp/lhagent.tar.gz
docker exec TASK sh -c 'mkdir -p /opt && tar -xzf /tmp/lhagent.tar.gz -C /opt'
docker exec TASK /opt/lhagent/lhagent --bundle-check
```

解压即能运行，不需要 pip、uv、系统 Python、venv、联网安装或修改 PATH。
目标 `/opt/lhagent` 应不存在；升级时解压到新的目录，避免混合两个版本。
非 root 用户可解压到任意可写目录。可选安装命令链接：

```sh
docker exec TASK /opt/lhagent/install.sh
# 或指定可写的命令目录（不会自动修改 PATH）
docker exec TASK /opt/lhagent/install.sh /some/writable/bin
```

在 Dockerfile 中注入也可以：

```dockerfile
FROM your-task-image
ADD lhagent-0.51.0-linux-amd64.tar.gz /opt/
# 可选：RUN /opt/lhagent/install.sh
```

运行（模型配置和凭据由 benchmark 提供）：

```sh
docker exec -w /workspace TASK /opt/lhagent/lhagent \
  --config /workspace/lhagent.toml --instruction '完成任务'
```

配置最小示例，`cwd` 必须指向任务工作目录；省略 `tools` 会启用所有内置工具：

```toml
[coding_agent]
model = "your-model-id"
context_window = 128000
max_output_tokens = 4096
cwd = "/workspace"
```

通过任务容器环境提供 `LHAGENT_BASE_URL` 和 `LHAGENT_API_KEY`。会话日志位于
启动工作目录的 `.lhagent/sessions`。包不包含凭据或个人配置。

## 环境边界与兼容性

* 私有 CPython 和依赖位于包的 `runtime/`。启动器通过绝对路径和 `-I` 运行，
  不激活 venv，不修改 PATH/PYTHONPATH/PYTHONHOME/VIRTUAL_ENV。
* `bash` 在配置的任务 cwd 中执行，继承任务环境；其中 `python`、`node` 和编译器
  仍来自任务镜像。现有工具主动过滤 BASH_ENV，避免非交互 shell 执行启动脚本。
* `find`/`grep` worker 使用同一个私有解释器，并以 `-I` 忽略任务 Python 的导入配置。
* 支持对应架构的 Linux **glibc** 镜像；本版本以 Debian bookworm 为构建基线。
  不支持 Alpine/musl、Windows 或 macOS。旧 glibc 镜像需单独验证，不能保证任意镜像通用。
* 镜像需提供 `/bin/sh`、`/bin/bash`、正常挂载的 `/proc`、`readlink -f`
  和基本目录工具；安装解压还需要 tar/gzip。包不替任务镜像安装系统软件。
* HTTPS 还需要网络和可信 CA；常规 Python HTTP 客户端的 certifi CA 已随依赖打包。
* 包是私有运行时隔离，不是文件系统沙箱；agent 仍能查看容器中有权限访问的路径，
  包目录也可见。task 命令自行修改 PATH 或显式调用包内 Python 不在此隔离范围内。

`--bundle-check` 离线验证导入、真实 find/grep worker、bash、任务环境变量及 Python
命令解析，不调用模型。`manifest.json` 记录构建时 Python、libc 和依赖版本。

## 验证

```sh
./packaging/test.sh packaging/dist/lhagent-0.51.0-linux-amd64.tar.gz linux/amd64
```

测试在临时容器里验证重定位、非 root/只读运行、无系统 Python 的环境，以及任务
Python 3.11 与私有 Python 3.12 的分离；不使用真实凭据。两种测试镜像均不安装
procps，Linux 进程组清理直接读取 `/proc`，不依赖任务镜像内的 `ps` 命令。
