FROM ghcr.io/astral-sh/uv:0.12.7 AS uv
FROM debian:bookworm-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends procps ca-certificates && rm -r /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /source
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY packaging ./packaging
ARG TARGETARCH
RUN bash packaging/build-linux.sh "$TARGETARCH"
FROM scratch AS bundle
COPY --from=builder /output/ /
