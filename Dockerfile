FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY sql ./sql
COPY src ./src
RUN pip install --no-cache-dir .
RUN useradd --create-home appuser
USER appuser
ENV NEM_LAB_ROOT=/data
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "nembattery.api:app", "--host", "0.0.0.0", "--port", "8000"]
