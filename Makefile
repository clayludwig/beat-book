.PHONY: install run dev lint clean help

PYTHON     := python3
VENV       := .venv
VENV_PY    := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
UVICORN    := $(VENV)/bin/uvicorn
HOST       := 127.0.0.1
PORT       := 8000

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install   Create venv and install dependencies"
	@echo "  run       Start the server (production-style, single worker)"
	@echo "  dev       Start the server with auto-reload (development)"
	@echo "  lint      Run pyflakes on all .py files"
	@echo "  clean     Remove venv, __pycache__, and .cache"

# ── Setup ─────────────────────────────────────────────────────────────────────

install: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $(VENV)/bin/activate
	@echo "✓ Environment ready. Run: make dev"

# ── Server ────────────────────────────────────────────────────────────────────

run: $(VENV)/bin/activate
	$(UVICORN) app:app --host $(HOST) --port $(PORT)

dev: $(VENV)/bin/activate
	$(UVICORN) app:app --host $(HOST) --port $(PORT) --reload

# ── Quality ───────────────────────────────────────────────────────────────────

lint: $(VENV)/bin/activate
	$(VENV)/bin/python -m pyflakes *.py

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	rm -rf $(VENV) __pycache__ .cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
