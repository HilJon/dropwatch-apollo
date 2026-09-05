# Installation

## Requirements

- Windows acquisition PC
- FastEye RLE camera
- Cypress USB camera driver
- Python 3.10 or newer

The vendor DLLs and their configuration bundle are included under
`dropwatch_apollo/data/core`. The built wheel is tagged `win_amd64`; a 64-bit Python
installation is required.

## Install

For an existing recorder environment, follow [MIGRATION.md](MIGRATION.md) first.
Uninstall `dropwatch-recorder` **before** installing this version: both provide
the `recorder` import namespace. Do not install them together.

```powershell
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install .
```

For parallel evaluation callbacks returning pandas DataFrames:

```powershell
pip install ".[evaluation]"
```

For development:

```powershell
pip install -e . --group dev
python -m pytest --cov=dropwatch_apollo --cov-fail-under=80
python -m ruff check .
python -m ruff format --check .
python -m mypy dropwatch_apollo recorder
```

No camera is needed for the software tests. Two real-decoder tests require
Windows and run against the bundled DLL; they are skipped on macOS/Linux.
Source-based development and pure-Python RLE replay work on those platforms,
but the distributable vendor-DLL wheel intentionally targets Windows x64.

The CLI is `dwa --help` or `python -m dropwatch_apollo --help`. For example:

```powershell
dwa record --frames 1000 --pre-trigger 20 --shots 40 --duration 60 --auto-trigger
```

The CLI saves lossless NPY shots to a unique directory under
`dropwatch_recordings/raw`, then exports AVI files. Use a fast local SSD for large
recordings. A connected camera is required for `HARDWARE_ACCEPTANCE.md`.

For isolated camera/USB diagnostics, run `dwa diagnose --help` and follow
[DIAGNOSTICS.md](DIAGNOSTICS.md). This command is available from version 0.2.1.
Use **0.2.3 or later** for hardware testing; it includes the startup fix, reduced
status polling and per-query diagnostic timing. Hardware qualification is still required.
