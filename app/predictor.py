import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Carga el pipeline serializado UNA vez y lo reutiliza en cada request."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self._model_dir = model_dir or settings.model_dir
        self._pipeline = None
        self._metadata: dict = {}

    def load(self) -> None:
        model_path = self._model_dir / settings.model_filename
        metadata_path = self._model_dir / settings.metadata_filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el artefacto del modelo en {model_path}. "
                "Corre training/train.py antes de levantar la API."
            )

        logger.info("Cargando modelo desde %s", model_path)
        self._pipeline = joblib.load(model_path)

        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                self._metadata = json.load(f)
        else:
            logger.warning("No se encontró metadata.json en %s", self._model_dir)

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def metadata(self) -> dict:
        return self._metadata

    def predict_one(self, features: dict, threshold: float = 0.5) -> tuple[float, bool]:
        if self._pipeline is None:
            raise RuntimeError("El modelo no está cargado. Llama a load() primero.")

        df = pd.DataFrame([features])
        proba = float(self._pipeline.predict_proba(df)[0, 1])
        return proba, proba >= threshold

    def predict_many(self, instances: list[dict]) -> list[tuple[float, bool]]:
        if self._pipeline is None:
            raise RuntimeError("El modelo no está cargado. Llama a load() primero.")

        thresholds = [inst.get("threshold", 0.5) for inst in instances]
        df = pd.DataFrame(instances)
        probas = self._pipeline.predict_proba(df)[:, 1]
        return [(float(p), bool(p >= t)) for p, t in zip(probas, thresholds, strict=True)]


predictor = ChurnPredictor()
