.PHONY: setup validate test migrate run db-cleanup

setup:
	python -m app.cli setup

validate:
	python -m app.cli config validate

test:
	python -m compileall -q app migrations tests
	pytest -q

migrate:
	python -m app.cli db migrate

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

db-cleanup:
	python -m app.cli db cleanup
