FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app

COPY pyproject.toml uv.lock setup.py README.md CHANGELOG.md LICENCE ./
COPY xirt/ xirt/
RUN uv sync --locked --no-dev --no-editable --extra cpu --no-cache \
    && useradd --create-home xirt \
    && mkdir /data \
    && chown xirt:xirt /data

USER xirt
WORKDIR /data
ENTRYPOINT ["xirt"]
