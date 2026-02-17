Contributing
============

Developer setup
---------------

1. Create a virtual environment and activate it (recommended):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install project and dev dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Install pre-commit hooks (once):

```bash
python -m pre_commit install
```

Usage
-----

- Run the pre-commit hooks on all files:

```bash
python -m pre_commit run --all-files
```

- Run the test suite:

```bash
python -m pytest -q
```

- Run a quick smoke example (small, quick):

```bash
python scripts/smoke_run.py
```

Notes
-----
- `black` enforces code formatting; `isort` sorts imports; `mypy` checks types.
- CI runs `mypy` and `pytest` on push/PR.
