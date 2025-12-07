# базовый образ python 3.10 slim 
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir --upgrade pip build
RUN python3 -m build --wheel
FROM python:3.10-slim AS runner
WORKDIR /app
COPY --from=builder /app/dist /app/dist
COPY --from=builder /app/tests /app/tests
RUN pip install --no-cache-dir numpy /app/dist/*.whl
CMD ["python3", "tests/check_mac.py"]