FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BRICKDNN_DATA_DIR=/data

WORKDIR /app

COPY requirements.txt .
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

CMD ["sh", "-c", "python manage.py migrate --noinput && exec gunicorn brick_dnn_site.wsgi:application --bind 0.0.0.0:8000 --workers ${GUNICORN_WORKERS:-2} --timeout ${GUNICORN_TIMEOUT:-300} --access-logfile -"]
