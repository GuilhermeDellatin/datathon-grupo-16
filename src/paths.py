"""Helpers para resolver caminhos relativos à raiz do projeto."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(*parts: str) -> Path:
    """Retorna um caminho absoluto a partir da raiz do repositório."""
    return PROJECT_ROOT.joinpath(*parts)


def resolve_project_file(path_like: str | Path) -> Path:
    """Resolve caminho absoluto preservando caminhos já absolutos."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return project_path(*path.parts)
