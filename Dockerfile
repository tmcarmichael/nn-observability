FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip curl git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy project
WORKDIR /workspace
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY scripts/ scripts/

# Install all deps including transformer extras
RUN uv sync --extra transformer

# Default: show available models
CMD ["uv", "run", "python", "scripts/run_model.py", "--help"]
