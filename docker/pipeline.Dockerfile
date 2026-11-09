# syntax=docker/dockerfile:1
# Batch job: data collection + statistical/ML analysis + static-site rendering.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    DATA_DIR=/work/data OUT_DIR=/work/out SITE_DIR=/work/site HISTORY_DIR=/work/history \
    HF_HOME=/work/cache/hf MPLCONFIGDIR=/tmp/mpl NUMBA_CACHE_DIR=/tmp/numba OMP_NUM_THREADS=4

RUN apt-get update \
 && apt-get install -y --no-install-recommends tini \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r /tmp/requirements.txt

RUN groupadd -g 10001 app && useradd -u 10001 -g app -M -s /usr/sbin/nologin app \
 && mkdir -p /work && chown app:app /work

WORKDIR /app
COPY src/ /app/src/
USER app
ENTRYPOINT ["/usr/bin/tini", "--", "python", "-m", "aitrends"]
CMD ["all"]
