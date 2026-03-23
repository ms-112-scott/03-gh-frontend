"""Frontend adapters for the 02-nca-cfd inference interface."""

from src.integration.nca_engine import NCAEngine, PhysicalConditions
from src.integration.nca_loader import PipelineInfo, load_inference_pipeline

__all__ = [
    "NCAEngine",
    "PhysicalConditions",
    "PipelineInfo",
    "load_inference_pipeline",
]
