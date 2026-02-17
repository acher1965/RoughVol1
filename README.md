# RoughVol1

Rough volatility experiments and rBergomi-based simulation and calibration.

Note: Version 1.1 is the last human-only version.
The canonical version is stored in [RoughVol1/__init__.py](RoughVol1/__init__.py) as `__version__`.

## What this does
- Simulates rBergomi paths and prices options.
- Computes implied vol surfaces and distances to market surfaces.
- Supports bulk runs from spreadsheet inputs and writes results to Excel.

## Project layout
- Core logic: [RoughVol1/RFSV_functions.py](RoughVol1/RFSV_functions.py)
- Helpers and I/O: [RoughVol1/RFSV_helpers.py](RoughVol1/RFSV_helpers.py)
- Entry point: [RoughVol1/RFSV_main.py](RoughVol1/RFSV_main.py)
- Error metrics: [RoughVol1/RFSV_functions_Error.py](RoughVol1/RFSV_functions_Error.py)

## Requirements
- Python 3.10+ (tested with Python 3.12)
- Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quick start
Run the main script (expects input files in the configured location):

```bash
python RoughVol1/RFSV_main.py --log-level INFO
```

## Input spreadsheet expectations
The default input file is `RFSV_input.xlsx` (override with `-x`). It must include:

- A main sheet with the request table (read with `header=0`, `skiprows=1`).
- Columns for shared configuration and per-request parameters.
- Tenor and forward/variance series columns.
- Two additional sheets: `VolGrid` and `WeightsGrid`.

Minimum expected columns on the main sheet:

- `Destination` (output folder prefix)
- `Row`, `Date`, `Spot`, `Underlying`
- `Simulations`, `Seed`
- `Roughness(Hurst)`, `VolVol(Eta)`, `Correlation(Rho)`
- `FullDiagnostic`, `FullOutput`
- `Tenor1..TenorN`
- `Fwd1..FwdN`
- `Impvar1..ImpvarN`

Row selection notes:
- Positional arguments are 1-based row numbers.
- Passing `0` runs the last row in the sheet.

## Example workflow
1. Prepare an input Excel file with the columns and sheets above.
2. Run a single request row:

```bash
python RoughVol1/RFSV_main.py -x RFSV_input.xlsx 1
```

3. Run multiple rows and write outputs to the `Destination` path:

```bash
python RoughVol1/RFSV_main.py -f C:\path\to\inputs -x RFSV_input.xlsx 1 2 3
```

## Tests
Run unit tests:

```bash
python -m pytest -q
```

Run type checks:

```bash
python -m mypy --config-file mypy.ini RoughVol1
```

## Formatting and hooks
Pre-commit hooks are configured for `black`, `isort`, and `mypy`.

```bash
python -m pre_commit install
python -m pre_commit run --all-files
```

On Windows, if you see long-path errors during hook setup, set a short cache:

```bash
setx PRE_COMMIT_HOME C:\pccache
```

## Version notes
- 1.0.0: brute-force rBergomi simulation.
- 1.1.0: improved inputs/outputs, bulk runs, and distance metrics.
- 1.1 is the last human-only version.
