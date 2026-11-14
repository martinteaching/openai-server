FROM python:latest
ARG UID=1000
ARG GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid $GID app && useradd --uid $UID --gid $GID --create-home app \
    && mkdir -p /openai-server/output && chown -R app:app /openai-server
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

USER app
WORKDIR /openai-server
RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/home/app/.local/bin:${PATH}"

COPY --chown=app:app pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra quant
COPY --chown=app:app . .
RUN uv sync --frozen --no-dev --extra quant

ENV LOGLEVEL=DEBUG
CMD ["uv", "run", "--no-sync", "openaiserver"]
