from __future__ import annotations

from .graph import WorkflowRuntime
from .schemas import PipelineRequest
from .settings import Settings


def build_tools() -> list[object]:
    try:
        from langchain_core.tools import StructuredTool  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("langchain-core is required for LangChain tool adapter.") from exc

    runtime = WorkflowRuntime(Settings.from_env())

    def robotic_sorting_pipeline(image_path: str, instruction: str) -> dict:
        req = PipelineRequest.model_validate({"image": {"image_path": image_path}, "instruction": instruction})
        return runtime.invoke(req).model_dump()

    tool = StructuredTool.from_function(
        func=robotic_sorting_pipeline,
        name="robotic_sorting_pipeline",
        description="Run the end-to-end robotic semantic sorting pipeline.",
    )
    return [tool]
