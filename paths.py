from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))

_legacy_user = os.getenv("LEGACY_FVLM_USER", "muhammedg")
LEGACY_PROJECT_ROOT = str(Path("/home") / _legacy_user / "fvlm")


def project_root() -> Path:
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parent


def data_root() -> Path:
    return Path(os.getenv("DATA_ROOT", str(project_root() / "data"))).expanduser().resolve()


def data_sym_root() -> Path:
    return Path(os.getenv("DATA_SYM_ROOT", str(project_root() / "data_sym"))).expanduser().resolve()


def checkpoints_root() -> Path:
    return Path(
        os.getenv("CHECKPOINT_ROOT", str(project_root() / "checkpoints"))
    ).expanduser().resolve()


def results_root() -> Path:
    return Path(os.getenv("RESULTS_ROOT", str(project_root() / "results"))).expanduser().resolve()


def resolve_legacy_path(path_value: str) -> str:
    value = os.path.expanduser(path_value)
    if value.startswith(LEGACY_PROJECT_ROOT):
        suffix = value[len(LEGACY_PROJECT_ROOT) :].lstrip("/")
        return str((project_root() / suffix).resolve())
    return value
