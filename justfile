export NIX_CONFIG := "extra-experimental-features = nix-command flakes"

default: lint fmt
all: lint fmt check

fmt:
    treefmt || nix fmt

lint:
    uv run ruff check --fix

test:
    uv run pytest src

check:
    uv run basedpyright --project pyproject.toml

sync:
    uv sync --all-packages

sync-clean:
    uv sync --all-packages --force-reinstall --no-cache

rust-rebuild:
    cargo run --bin stub_gen
    uv sync --reinstall-package exo_pyo3_bindings

build-dashboard:
    #!/usr/bin/env bash
    cd dashboard-react
    npm install
    npm run build

package:
    uv run pyinstaller packaging/pyinstaller/exo.spec

clean:
    rm -rf **/__pycache__
    rm -rf target/
    rm -rf .venv
    rm -rf dashboard-react/node_modules
    rm -rf dashboard-react/dist
