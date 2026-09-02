# Development

## Requirements

- Python 3.11 or 3.12 (uv will download one if needed)
- [just](https://github.com/casey/just)
- [uv](https://docs.astral.sh/uv/) and FFmpeg (`just install` will install them if missing)

There is no pip requirements file or Makefile. Dependencies live in `pyproject.toml` / `uv.lock`; recipes live in the `Justfile`.

## Setup

```bash
just install                 # clean + uv + FFmpeg + uv sync --all-groups
uv run pre-commit install
```

## Package layout

```text
whisper_subtitler/
  main.py
  version.py
  modules/
    application.py
    cli.py
    config.py
    gui/          # optional extra: pyside6
    runtime.py    # frozen exe helpers
    audio/
    diarisation/
    output/
    transcribe/
tests/
docs/
Justfile
packaging/        # PyInstaller spec + freeze script
pyproject.toml
```

Import as `whisper_subtitler.modules…`. The console script is:

```text
whisper-subtitler = whisper_subtitler.modules.cli:main
```

## Just recipes

| Recipe | Purpose |
|--------|---------|
| `just install` | Clean, install uv and FFmpeg, then `uv sync --all-groups` |
| `just install-gui` | Same as `just install`, plus the PySide6 GUI extra |
| `just install-uv` | Install uv if it is not on `PATH` |
| `just install-ffmpeg` | Install FFmpeg if it is not on `PATH` |
| `just sync` | Install all dependency groups |
| `just sync-gui` | `uv sync --all-groups --extra gui` |
| `just lock` | Refresh the lockfile |
| `just test` | Run pytest |
| `just lint` | Ruff check + format check |
| `just fmt` | Auto-fix with Ruff |
| `just typecheck` | basedpyright |
| `just check` | lint + typecheck + tests |
| `just run *args` | `uv run whisper-subtitler …` |
| `just gui` | Open the desktop GUI |
| `just dist` | Freeze onedir zip (`--extra gui --extra packaging`; CPU torch on Linux/Windows). Re-run `just sync` afterwards if you need CUDA wheels in the venv. |
| `just docs` | `mkdocs build -s` |
| `just clean` | Remove caches, build artifacts, and `.venv` |

## Tests and tox

```bash
just test
just check
uv run tox    # py311, py312, py313
```

## Docs

Published pages are under `docs/` (see `mkdocs.yml`). Historical Memory Bank / design notes live in `docs/archive/` and are not the source of truth.

```bash
just docs
```

## Version

The package version is defined once in `whisper_subtitler/version.py` and exposed via:

```bash
uv run whisper-subtitler version
```
