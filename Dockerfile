# GroundedRx — on-premises deployment image.
#
# Single container, no CPU/GPU microservice split -- unnecessary complexity
# for what this is. Model weights are NOT baked into the image (first run
# downloads them via the Hugging Face cache, then it's fully self-contained
# and can run air-gapped); the vector store is NOT baked in either, it's
# mounted as a volume at runtime. See README "On-Premises Deployment".
#
# Base image ships Python + torch + CUDA already wired together correctly --
# reinstalling that by hand from a bare CUDA image is exactly the kind of
# dependency-version tightrope that broke this project's SGLang experiment
# (see CLAUDE.md "Backend history"). Reuse the known-good combination instead.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY pyproject.toml README.md ./
COPY groundedrx/ groundedrx/
RUN pip install --no-cache-dir -e ".[gpu,api]"

# Reduces OOM from memory fragmentation across many small generate() calls
# -- same mitigation used in the notebook's Setup cell.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Vector store: mount your extracted qdrant_db_archive/ here at runtime,
# e.g. `docker run -v ./qdrant_db_archive:/data/qdrant_storage:ro ...`.
# GROUNDEDRX_QDRANT_PATH short-circuits the Colab/Kaggle auto-detection in
# groundedrx/paths.py, which would otherwise fail looking for paths that
# don't exist inside a container.
ENV GROUNDEDRX_QDRANT_PATH=/data/qdrant_storage

EXPOSE 8000

CMD ["uvicorn", "groundedrx.api:app", "--host", "0.0.0.0", "--port", "8000"]
