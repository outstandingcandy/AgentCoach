# AgentCore Runtime image for the GoalInsight chat agent.
#
# Hard requirements per AgentCore:
#   - linux/arm64
#   - non-root user
#   - port 8080
#   - /invocations + /ping endpoints (see main.py)
#
# Build & push: deploy/agentcore_runtime/build_and_push.sh

FROM --platform=linux/arm64 python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY chat_app/ ./chat_app/
COPY main.py ./main.py

RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
