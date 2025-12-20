.PHONY: help install dev clean test lint format typecheck docs build publish

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in editable mode
	pip install -e .

dev: ## Install package with development dependencies
	pip install -e ".[dev,all]"

clean: ## Remove build artifacts and cache files
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=rag_toolkit --cov-report=html --cov-report=term

lint: ## Run linting (ruff)
	ruff check src/rag_toolkit tests

format: ## Format code (black + isort)
	black src/rag_toolkit tests examples scripts
	isort src/rag_toolkit tests examples scripts

format-check: ## Check code formatting without changes
	black --check src/rag_toolkit tests examples scripts
	isort --check-only src/rag_toolkit tests examples scripts

typecheck: ## Run type checking (mypy)
	mypy src/rag_toolkit

check: format-check lint typecheck test ## Run all checks

docs: ## Build documentation
	cd docs && make html

docs-serve: docs ## Build and serve documentation
	cd docs/build/html && python -m http.server 8000

build: clean ## Build distribution packages
	python -m build

publish-test: build ## Publish to TestPyPI
	twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	twine upload dist/*

fix-imports: ## Fix src. imports to rag_toolkit.
	python scripts/fix_imports.py --root=src/rag-toolkit

# Docker targets
docker-milvus: ## Start Milvus in Docker
	docker run -d --name milvus-standalone \
		-p 19530:19530 -p 9091:9091 \
		-v $${PWD}/volumes/milvus:/var/lib/milvus \
		milvusdb/milvus:latest

docker-milvus-stop: ## Stop Milvus container
	docker stop milvus-standalone
	docker rm milvus-standalone

docker-ollama: ## Start Ollama in Docker
	docker run -d --name ollama \
		-p 11434:11434 \
		-v $${PWD}/volumes/ollama:/root/.ollama \
		ollama/ollama:latest

docker-ollama-stop: ## Stop Ollama container
	docker stop ollama
	docker rm ollama

# Development workflows
dev-setup: dev docker-milvus docker-ollama ## Complete development setup
	@echo "✅ Development environment ready!"
	@echo "   - Milvus running on http://localhost:19530"
	@echo "   - Ollama running on http://localhost:11434"
	@echo ""
	@echo "Next steps:"
	@echo "   1. Pull models: docker exec ollama ollama pull nomic-embed-text"
	@echo "   2. Pull LLM: docker exec ollama ollama pull llama3.2"
	@echo "   3. Run tests: make test"
	@echo "   4. Try example: python examples/quickstart.py"

dev-teardown: docker-milvus-stop docker-ollama-stop ## Stop all development services
	@echo "✅ Development services stopped"
