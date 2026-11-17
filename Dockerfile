# CPU-only image that regenerates the possession CSV from the organisers' bundle.
#
#   docker build -t gdkb-submission .
#   docker run --rm -v /path/to/bundle:/data:ro -v "$PWD/out":/out gdkb-submission /data /out/<team_name>.csv
#
# The bundle is mounted at run time and never copied into the image (RULES.md §"The data").
FROM python:3.12.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PIP_NO_CACHE_DIR=1

# libgomp is LightGBM's only system dependency on the manylinux wheel (torch CPU wheel needs nothing else)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes -r /app/requirements.txt

COPY gdkb /app/gdkb
COPY models /app/models
COPY predict.py validate.py /app/

ENTRYPOINT ["python", "/app/predict.py"]
