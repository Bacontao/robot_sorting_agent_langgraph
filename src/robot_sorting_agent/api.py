from __future__ import annotations

from fastapi import FastAPI

from .graph import WorkflowRuntime
from .schemas import PipelineRequest, PipelineResponse
from .settings import Settings
from .observability import configure_logging


def create_app() -> FastAPI:
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    runtime = WorkflowRuntime(settings)
    app = FastAPI(title="Robot Sorting Agent LangGraph")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/pipeline", response_model=PipelineResponse)
    def pipeline(request: PipelineRequest) -> PipelineResponse:
        return runtime.invoke(request)

    return app
