# Raccourcis du projet. Sous Windows sans `make`, les commandes de chaque
# cible se recopient telles quelles dans un terminal.

PYTHON ?= python

.PHONY: help install repro force test lint fmt metrics clean

help:  ## Liste les cibles disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Installe les dépendances d'exécution et de développement
	$(PYTHON) -m pip install -r requirements-dev.txt

repro:  ## Rejoue le pipeline (seules les étapes impactées sont recalculées)
	dvc repro

force:  ## Rejoue l'intégralité du pipeline, cache ignoré
	dvc repro --force

metrics:  ## Affiche les métriques et leur évolution depuis le dernier commit
	dvc metrics show
	@echo ""
	dvc metrics diff

test:  ## Lance la suite de tests
	$(PYTHON) -m pytest -q

lint:  ## Contrôle le style et les erreurs statiques
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

fmt:  ## Reformate le code
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

clean:  ## Supprime les caches Python
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
