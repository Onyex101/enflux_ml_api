from pydantic import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    app_name: str = "Enflux AI API"
    created_by: str = "Onyekachi A. Okoye"
    version: str = None
    description: str = "Enflux AI API for classifying questions based on Bloom's taxonomy"
    SECRET_KEY: str = None
    RDS_URL: str = None
    # RDS_USERNAME: str = None
    # RDS_PASSWORD: str = None
    # RDS_HOSTNAME: str = None
    # RDS_PORT: str = None
    # RDS_DB_NAME: str = None

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()