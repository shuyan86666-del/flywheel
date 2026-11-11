FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY flywheel.py sample_10.jsonl sample_10.说明.md ./

ENTRYPOINT ["python", "flywheel.py"]
CMD ["sample_10.jsonl", "--output", "/app/output/processed_10.jsonl"]
