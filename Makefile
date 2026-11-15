.PHONY: install train api app test lint format

install:
	pip install -r requirements-dev.txt

train:
	python -m retail_sales.train

api:
	flask --app "api.app:create_app()" run --port 7860 --debug

app:
	streamlit run app/streamlit_app.py

test:
	pytest

lint:
	ruff check . && ruff format --check .

format:
	ruff format . && ruff check --fix .
