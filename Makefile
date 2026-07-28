# Every target runs through uv, so the pinned lock file is the only environment
# that is ever exercised - locally and in CI alike.
#
# Targets appear here once the thing they drive exists. Compose targets arrive with
# docker-compose.yml, integration tests once there is an integration test to run.

.PHONY: install lint fmt types arch test check up down logs ps reset

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

# --- local data services ---------------------------------------------------------

up:
	docker compose up -d --wait

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

# Destroys the postgres and qdrant volumes. Guarded on purpose: it is one word away
# from every other target here.
reset:
	@test "$(CONFIRM)" = "yes" || { \
	  echo "make reset deletes all local database volumes."; \
	  echo "Re-run as: make reset CONFIRM=yes"; exit 1; }
	docker compose down -v
