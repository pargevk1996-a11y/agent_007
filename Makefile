# Every target runs through uv, so the pinned lock file is the only environment
# that is ever exercised - locally and in CI alike.
#
# Targets appear here once the thing they drive exists. Compose targets arrive with
# docker-compose.yml, integration tests once there is an integration test to run.

.PHONY: install lint fmt types arch test check

install:
	uv sync --all-groups

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

types:
	uv run mypy researchmind tests

arch:
	uv run lint-imports

test:
	uv run pytest -m "not integration and not e2e"

check: lint types arch test
