# One image for the app and every MCP server (CPU only; no GPU needed).
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src HF_HOME=/models TOKENIZERS_PARALLELISM=false
WORKDIR /app
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.* && pip install -r requirements.txt
# Bake the embedding + reranker models into the image so containers start offline.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
COPY configs ./configs
COPY corpus ./corpus
COPY src ./src
# eval/heldout is deliberately NOT copied: agents in the container cannot read held-out cases.
RUN useradd -m desk && mkdir -p /data && chown desk /data
USER desk
ENV AGENT_DESK_DATA=/data
EXPOSE 8000
CMD ["uvicorn", "agent_desk.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
