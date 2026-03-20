ARG PYTHON_IMAGE=python:3.11-slim

FROM ${PYTHON_IMAGE} AS builder

WORKDIR /src
COPY pyproject.toml README.md ./
COPY flashrl ./flashrl
RUN python -m pip install --upgrade pip build
RUN python -m build --wheel

FROM ${PYTHON_IMAGE}

WORKDIR /app
COPY --from=builder /src/dist/*.whl /tmp/dist/
RUN python -m pip install --upgrade pip \
    && python -m pip install /tmp/dist/*.whl "kubernetes>=32.0.0" \
    && rm -rf /tmp/dist

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["flashrl"]
CMD ["component", "run", "controller"]
