# Convenience commands. Run `make help` for an overview.
PYTHON ?= python3
VENV = backend/.venv/bin

.PHONY: help install backend frontend test lint docker

help:
	@echo "make install   create backend venv and install all dependencies"
	@echo "make backend   run the API on http://127.0.0.1:8000"
	@echo "make frontend  run the UI on http://localhost:5173"
	@echo "make test      run all tests"
	@echo "make lint      lint the backend"
	@echo "make docker    build and start everything with Docker Compose"

install:
	cd backend && $(PYTHON) -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
	cd frontend && npm install

backend:
	cd backend && .venv/bin/python manage.py migrate && DJANGO_DEBUG=true .venv/bin/python manage.py runserver

frontend:
	cd frontend && npm run dev

test:
	cd backend && .venv/bin/python -m pytest
	cd frontend && npm test

lint:
	cd backend && .venv/bin/ruff check . && .venv/bin/ruff format --check .

docker:
	docker compose up -d --build
