.PHONY: help build up down migrate shell test clean

help:
	@echo "Available commands:"
	@echo "  make build    - Build Docker images"
	@echo "  make up       - Start services"
	@echo "  make down     - Stop services"
	@echo "  make migrate  - Run database migrations (auto-run on start)"
	@echo "  make shell    - Open Django shell"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Clean up containers and volumes"

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

migrate:
	docker compose exec web python manage.py migrate

shell:
	docker compose exec web python manage.py shell

test:
	docker compose exec web python manage.py test

clean:
	docker compose down -v
	docker system prune -f

