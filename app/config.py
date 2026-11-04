"""Runtime settings: where the database and the trained model live."""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = ROOT / "data" / "aerosentry.db"
DEFAULT_MODEL_PATH = ROOT / "models" / "rul_model.joblib"


@dataclass(frozen=True)
class Settings:
    db_path: Path
    model_path: Path


def get_settings() -> Settings:
    """Read settings from the environment, falling back to the repository layout.

    AEROSENTRY_DB and AEROSENTRY_MODEL override the default paths. Tests use this
    to point the app at small temporary files.
    """
    return Settings(
        db_path=Path(os.environ.get("AEROSENTRY_DB", DEFAULT_DB_PATH)),
        model_path=Path(os.environ.get("AEROSENTRY_MODEL", DEFAULT_MODEL_PATH)),
    )
