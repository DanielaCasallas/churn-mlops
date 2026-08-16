from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChurnPredictRequest(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(..., description="1 si es adulto mayor, 0 si no")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=100, description="Meses como cliente")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0, le=500)
    TotalCharges: float = Field(..., ge=0, le=20000)
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Umbral de decisión para clasificar churn")

    @field_validator("TotalCharges")
    @classmethod
    def total_at_least_monthly_times_zero(cls, v: float) -> float:
        # TotalCharges no debería ser negativo; ya lo cubre `ge=0`, se deja como
        # ejemplo explícito de validación adicional sobre el dominio del problema.
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
                "threshold": 0.5,
            }
        }
    }


class ChurnPredictResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    threshold_used: float


class BatchPredictRequest(BaseModel):
    instances: list[ChurnPredictRequest] = Field(..., min_length=1, max_length=500)


class BatchPredictResponse(BaseModel):
    predictions: list[ChurnPredictResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_type: str | None = None


class ModelSchemaResponse(BaseModel):
    features: dict
    target: str
    metrics: dict
    trained_at: str
