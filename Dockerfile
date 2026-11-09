# Stage 1: build the database and the model from raw data (reproducible, NFR-6).
# The model is trained here, never when the container starts.
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pipeline/ pipeline/
COPY data/raw/ data/raw/
RUN mkdir -p models \
    && python -m pipeline.load_db \
    && python -m pipeline.train

# Stage 2: runtime image. Non-root user (NFR-5).
FROM python:3.11-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY pipeline/ pipeline/
COPY --from=builder /build/models/ models/
COPY --from=builder /build/data/aerosentry.db data/aerosentry.db
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /srv
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
