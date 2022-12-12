from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
import models
from database import engine
from routers import auth, predictions, users
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from config.config import get_settings

app = FastAPI()

config = get_settings()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models.Base.metadata.create_all(bind=engine)

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(predictions.router)

@app.get("/")
async def redirect_typer():
    return RedirectResponse("/docs")

@app.get('/api/healthchecker')
def root():
    return {
        "status": 200,
        "title": config.app_name,
        "version": config.version,
        "description": config.description
    }


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=config.app_name,
        version=config.version,
        description=config.description,
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://enflux-static-prod.s3.amazonaws.com/email/Logo_signature.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.exception_handler(500)
async def internal_exception_handler(request: Request, exc: Exception):
  return JSONResponse(status_code=500, content=jsonable_encoder({"code": 500, "detail": "Internal Server Error"}))
