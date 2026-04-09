from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.services.inference import get_inference_status

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
INDEX_FILE = FRONTEND_DIR / "index.html"

app = FastAPI(
    title="TCAI API",
    description="Turtle Care AI demo backend with a mock QA endpoint.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


@app.get("/")
async def serve_frontend() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    status = get_inference_status()
    return {"status": "ok", **status}
