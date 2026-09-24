FROM debian:bookworm-slim AS no-python
FROM no-python AS task-python
RUN apt-get update && apt-get install -y --no-install-recommends python3 python-is-python3 && rm -r /var/lib/apt/lists/*
