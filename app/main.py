import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import settings
from app.predictor import predictor
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ChurnPredictRequest,
    ChurnPredictResponse,
    HealthResponse,
    ModelSchemaResponse,
)

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Se carga el modelo UNA sola vez al arrancar, no en cada request.
    try:
        predictor.load()
        logger.info("Modelo cargado correctamente.")
    except FileNotFoundError as exc:
        # No tumbamos el proceso: /health reportará model_loaded=False
        # y /predict devolverá 503 hasta que exista el artefacto.
        logger.error("No se pudo cargar el modelo al arrancar: %s", exc)
    yield


app = FastAPI(
    title="Churn Prediction Service",
    description="Servicio de predicción de riesgo de deserción de clientes (Telco Customer Churn).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    # Traduce los errores de Pydantic a un mensaje accionable: qué campo falló y por qué.
    errors = [
        {"field": ".".join(str(p) for p in e["loc"][1:]), "error": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": "Entrada inválida", "errors": errors})


def _require_model_loaded() -> None:
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Verifica que models/model.joblib exista.",
        )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        model_type=predictor.metadata.get("model_type") if predictor.is_loaded else None,
    )


@app.get("/model/schema", response_model=ModelSchemaResponse, tags=["ops"])
def model_schema() -> ModelSchemaResponse:
    _require_model_loaded()
    meta = predictor.metadata
    return ModelSchemaResponse(
        features=meta.get("features", {}),
        target=meta.get("target", "Churn"),
        metrics=meta.get("metrics", {}),
        trained_at=meta.get("trained_at", ""),
    )


@app.post("/predict", response_model=ChurnPredictResponse, tags=["inference"])
def predict(payload: ChurnPredictRequest) -> ChurnPredictResponse:
    _require_model_loaded()
    features = payload.model_dump(exclude={"threshold"})
    try:
        proba, is_churn = predictor.predict_one(features, threshold=payload.threshold)
    except Exception as exc:  # noqa: BLE001 - traducido a error accionable, nunca 500 genérico
        logger.exception("Error durante la inferencia")
        raise HTTPException(status_code=500, detail=f"Error al generar la predicción: {exc}") from exc

    return ChurnPredictResponse(
        churn_probability=round(proba, 4),
        churn_prediction=is_churn,
        threshold_used=payload.threshold,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    _require_model_loaded()
    instances = [inst.model_dump(exclude={"threshold"}) | {"threshold": inst.threshold} for inst in payload.instances]
    try:
        results = predictor.predict_many(instances)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error durante la inferencia batch")
        raise HTTPException(status_code=500, detail=f"Error al generar las predicciones: {exc}") from exc

    predictions = [
        ChurnPredictResponse(
            churn_probability=round(p, 4),
            churn_prediction=is_churn,
            threshold_used=inst.threshold,
        )
        for (p, is_churn), inst in zip(results, payload.instances, strict=True)
    ]
    return BatchPredictResponse(predictions=predictions)
