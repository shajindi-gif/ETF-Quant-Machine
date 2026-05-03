.PHONY: setup scan report test clean

setup:
	python3.12 -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

scan:
	. .venv/bin/activate && python main.py

report:
	. .venv/bin/activate && python main.py

test:
	. .venv/bin/activate && pytest -q

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
