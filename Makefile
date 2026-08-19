.PHONY: help sync docs-serve docs-serve-docker docs-build docs-figures docs-forward test lint format typecheck validate publish-testpypi-dev clean

DOCS_PORT := 7777

help:
	@echo "Available targets:"
	@echo "  sync          - uv sync (installs dev + docs dependency groups)"
	@echo "  docs-serve    - serve MkDocs site on 0.0.0.0:8000 (reachable over LAN/Tailscale)"
	@echo "  docs-serve-docker - serve MkDocs site in Docker on :8000 (auto-forwarded to Windows via Docker Desktop, no netsh needed)"
	@echo "  docs-build    - build MkDocs site with --strict (fails on broken links/warnings)"
	@echo "  docs-figures  - regenerate the real figures used in docs/theory/examples.md"
	@echo "  docs-forward  - print the Windows netsh portproxy command to reach docs-serve from Windows"
	@echo "  test          - run pytest"
	@echo "  lint          - run ruff check"
	@echo "  format        - run ruff format"
	@echo "  typecheck     - run ty check"
	@echo "  validate      - run itis-sumo's Dakota engine probe"
	@echo "  publish-testpypi-dev - create and upload next .devN release to TestPyPI"
	@echo "  clean         - remove build artifacts (site/, dist/, __pycache__)"

sync:
	uv sync --all-groups

docs-serve:
	uv run --group docs mkdocs serve -a 0.0.0.0:7777

# Serves via a container instead of a bare host process. Under WSL2, a
# plain `mkdocs serve` bound to 0.0.0.0 is not reliably reachable from a
# Windows browser without manual netsh portproxy (see docs-forward);
# Docker Desktop's WSL2 integration forwards published container ports to
# Windows localhost automatically.
docs-serve-docker:
	docker build -t itis-sumo-docs -f docker/Dockerfile.docs .
	docker rm -f itis-sumo-docs 2>/dev/null; \
	docker run -d --name itis-sumo-docs -p 8000:8000 -v $(CURDIR):/docs itis-sumo-docs
	@echo "Serving at http://localhost:8000/ (stop with: docker rm -f itis-sumo-docs)"

docs-build:
	uv run --group docs mkdocs build --strict

docs-figures:
	uv run --group docs python examples/generate_docs_figures.py

docs-forward:
	@wsl_ip=$$(hostname -I | awk '{print $$1}'); \
	echo "WSL IP: $$wsl_ip (changes on every WSL restart - rerun this if forwarding breaks)"; \
	echo ""; \
	echo "Run in Windows PowerShell as Administrator:"; \
	echo ""; \
	echo "  netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=$(DOCS_PORT)"; \
	echo "  netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=$(DOCS_PORT) connectaddress=$$wsl_ip connectport=$(DOCS_PORT)"; \
	echo ""; \
	echo "Verify with:  netsh interface portproxy show v4tov4"; \
	echo "Remove with:  netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=$(DOCS_PORT)"

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run ty check

validate:
	uv run itis-sumo validate


publish-testpypi-dev:
	@test -f .env || { echo "Create .env with TESTPYPI_TOKEN=pypi-..."; exit 1; }
	@test -z "$$(git status --porcelain)" || { echo "Commit or stash changes before publishing"; exit 1; }
	@set -e; \
	 backup=$$(mktemp); \
	 cp pyproject.toml "$$backup"; \
	 trap 'status=$$?; cp "$$backup" pyproject.toml; rm -f "$$backup"; exit "$$status"' EXIT; \
	 set -a; . ./.env; set +a; test -n "$$TESTPYPI_TOKEN" || { echo "Set TESTPYPI_TOKEN in .env"; exit 1; }; \
	 version=$$(uv run --no-project --with packaging python scripts/dev_version.py --write); \
	 echo "Publishing $$version to TestPyPI"; \
	 rm -rf dist/; \
	 uv build; \
	 uvx twine check dist/*; \
	 uv publish --index testpypi -t $$TESTPYPI_TOKEN

clean:
	rm -rf site/ dist/
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
