FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies (matching training sklearn version for joblib compatibility)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    scikit-learn==1.9.1 \
    pandas \
    joblib

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models && \
    chown -R appuser:appuser /app

COPY --chown=appuser:appuser api /app/api
COPY --chown=appuser:appuser models /app/models

USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
