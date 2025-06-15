"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
#
#
def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.
#
    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
#     """
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines

# from final.pipelines.data_preparation.pipeline import data_preparation
# def register_pipelines():
#     return {"data_processing": data_preparation.create_pipeline()}