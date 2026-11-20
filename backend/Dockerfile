# Debian 13 (trixie): its PostgreSQL client is version 17, the version of the db service in
# docker-compose.yml and newer than or equal to Supabase's, so that backups use pg_dump
# (brick_dnn_core/backups.py) and `manage.py move_database` works.
FROM python:3.12-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BRICKDNN_DATA_DIR=/data \
    PATH=/usr/lib/postgresql/17/bin:$PATH

# pg_dump, pg_restore and psql 17 (about 15 MB); libgomp1, the OpenMP runtime LightGBM needs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-client-17 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Includes scikit-learn (with SciPy, about 190 MB): classical models (Gaussian processes,
# SVMs) train in the default image, without PyTorch. XGBoost (CPU build), LightGBM and
# CatBoost add about 420 MB: CatBoost 270 MB, the plotting libraries it requires (matplotlib,
# plotly) about 110 MB, XGBoost 25 MB, LightGBM 10 MB.
RUN pip install --no-cache-dir -r requirements.txt

# Training needs PyTorch, which adds about 1 GB to the image. It is therefore optional:
#   BRICKDNN_INSTALL_TORCH=true docker compose up -d --build
# Docker only gets the CPU build; for GPU training run the backend outside Docker.
ARG INSTALL_TORCH=false
RUN if [ "$INSTALL_TORCH" = "true" ]; then \
      pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY . .
RUN DJANGO_SECRET_KEY=build-only python manage.py collectstatic --noinput \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /data && chown app /data

USER app
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health/')"

CMD ["sh", "-c", "python manage.py migrate --noinput && exec gunicorn brick_dnn_site.wsgi:application --bind 0.0.0.0:8000 --workers ${GUNICORN_WORKERS:-2} --timeout ${GUNICORN_TIMEOUT:-300} --access-logfile - --access-logformat '%(t)s \"%(m)s %(U)s %(H)s\" %(s)s %(B)s %(L)s'"]
