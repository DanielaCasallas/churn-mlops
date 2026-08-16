from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_dir: Path = Path(__file__).resolve().parent.parent / "models"
    model_filename: str = "model.joblib"
    metadata_filename: str = "metadata.json"
    log_level: str = "INFO"
    default_threshold: float = 0.5

    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env")


settings = Settings()
