.PHONY: help data data-small test run-als-lgbm run-tt-lgbm run-tt-neural run-all evaluate serve clean

PY ?= python3

help:
	@echo "Targets:"
	@echo "  make data-small     Download ml-latest-small (fast iteration)"
	@echo "  make data           Download ml-25m (full)"
	@echo "  make test           Run unit tests (metrics correctness)"
	@echo "  make run-als-lgbm   ALS retrieval -> LambdaMART ranker"
	@echo "  make run-tt-lgbm    Two-tower retrieval -> LambdaMART ranker"
	@echo "  make run-tt-neural  Two-tower retrieval -> neural ranker (listwise+pairwise)"
	@echo "  make run-all        Run every experiment config"
	@echo "  make evaluate       Aggregate results into results/metrics.csv + plots"
	@echo "  make serve          Launch FastAPI /rank endpoint"

data-small:
	$(PY) data/download.py --small

data:
	$(PY) data/download.py

test:
	$(PY) -m pytest tests/ -q

run-als-lgbm:
	$(PY) -m src.train --config configs/als_lambdamart.yaml

run-tt-lgbm:
	$(PY) -m src.train --config configs/two_tower_lambdamart.yaml

run-tt-neural:
	$(PY) -m src.train --config configs/two_tower_neural.yaml

run-all: run-als-lgbm run-tt-lgbm run-tt-neural

evaluate:
	$(PY) -m src.evaluate --results-dir results

serve:
	$(PY) -m uvicorn src.serve:app --host 0.0.0.0 --port 8000

clean:
	rm -rf artifacts/ results/*.csv results/*.png
