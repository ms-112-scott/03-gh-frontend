from __future__ import annotations

import importlib
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, TypedDict

import torch


logger = logging.getLogger("RhinoBridge")
FRONTEND_ROOT = Path(__file__).resolve().parents[2]


class PipelineInfo(TypedDict):
    pipeline: Any
    backend_root: str
    rans_checkpoint_dir: str
    inference_config_path: str | None


def _resolve_frontend_relative(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()

    candidates = [
        (FRONTEND_ROOT / "src" / path).resolve(),
        (FRONTEND_ROOT / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_checkpoint_file(path_value: str) -> Path:
    checkpoint_path = _resolve_frontend_relative(path_value)
    if checkpoint_path.exists():
        return checkpoint_path

    parent = checkpoint_path.parent
    if not parent.exists():
        return checkpoint_path

    preferred_matches = sorted(parent.glob(f"{checkpoint_path.stem.replace('_latest', '')}*.pth"))
    if preferred_matches:
        return preferred_matches[-1]

    generic_matches = sorted(parent.glob("*.pth"))
    if generic_matches:
        return generic_matches[-1]

    return checkpoint_path


def _resolve_repo_root_from_checkpoint(checkpoint_path: str) -> Path:
    ckpt_path = Path(checkpoint_path).resolve()
    for parent in [ckpt_path.parent, *ckpt_path.parents]:
        if parent.name == "02-nca-cfd":
            return parent
    raise FileNotFoundError(
        f"Unable to infer 02-nca-cfd root from checkpoint path: {checkpoint_path}"
    )


def _resolve_repo_root(frontend_config: Mapping[str, Any]) -> Path:
    nca_cfd_src = frontend_config.get("nca_cfd_src")
    if nca_cfd_src:
        src_path = _resolve_frontend_relative(str(nca_cfd_src))
        if src_path.name == "src":
            return src_path.parent
        return src_path

    checkpoint_path = frontend_config.get("nca_checkpoint")
    if checkpoint_path:
        return _resolve_repo_root_from_checkpoint(str(_resolve_checkpoint_file(str(checkpoint_path))))

    backend_root = frontend_config.get("nca_cfd_root")
    if backend_root:
        return Path(str(backend_root)).resolve()

    raise FileNotFoundError(
        "Missing backend location. Set one of: nca_cfd_src, nca_checkpoint, nca_cfd_root."
    )


def _inject_backend_repo(backend_root: Path) -> None:
    backend_root_str = str(backend_root)
    if backend_root_str not in sys.path:
        sys.path.insert(0, backend_root_str)
        logger.info("[Loader] sys.path += %s", backend_root_str)

    backend_src = backend_root / "src"
    if backend_src.is_dir():
        import src as frontend_src  # noqa: E402

        backend_src_str = str(backend_src)
        if backend_src_str not in frontend_src.__path__:
            frontend_src.__path__.append(backend_src_str)
            logger.info("[Loader] frontend src namespace += %s", backend_src_str)


def _resolve_rans_checkpoint_dir(frontend_config: Mapping[str, Any]) -> str:
    checkpoint_dir = frontend_config.get("rans_checkpoint_dir")
    if checkpoint_dir:
        return str(_resolve_frontend_relative(str(checkpoint_dir)))

    checkpoint_path = frontend_config.get("nca_checkpoint")
    if checkpoint_path:
        return str(_resolve_checkpoint_file(str(checkpoint_path)).parent)

    raise FileNotFoundError(
        "Missing RANS checkpoint location. Set rans_checkpoint_dir or nca_checkpoint."
    )


def load_inference_pipeline(frontend_config: Mapping[str, Any]) -> PipelineInfo:
    backend_root = _resolve_repo_root(frontend_config)
    _inject_backend_repo(backend_root)

    explicit_checkpoint: str | None = None
    if frontend_config.get("nca_checkpoint"):
        explicit_checkpoint = str(_resolve_checkpoint_file(str(frontend_config["nca_checkpoint"])))
        base_module = importlib.import_module("inference.BaseModuleEngine")
        original_find_checkpoint = base_module.BaseModuleEngine._find_checkpoint

        def _patched_find_checkpoint(ckpt_dir: str) -> str:
            if explicit_checkpoint is not None:
                explicit_path = Path(explicit_checkpoint).resolve()
                if explicit_path.parent == Path(ckpt_dir).resolve():
                    return str(explicit_path)
            return original_find_checkpoint(ckpt_dir)

        base_module.BaseModuleEngine._find_checkpoint = staticmethod(_patched_find_checkpoint)

    from inference import InferencePipeline  # noqa: E402

    inference_config_path = frontend_config.get("inference_config_path")
    runtime_device = frontend_config.get("device")
    if runtime_device is None:
        runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    if inference_config_path:
        inference_config_path = str(_resolve_frontend_relative(str(inference_config_path)))
        pipeline = InferencePipeline.from_config(inference_config_path)
        rans_checkpoint_dir = frontend_config.get("rans_checkpoint_dir", "<from-inference-config>")
    else:
        rans_checkpoint_dir = _resolve_rans_checkpoint_dir(frontend_config)
        candidate_kwargs = {
            "rans_checkpoint_dir": rans_checkpoint_dir,
            "turb_checkpoint_dir": frontend_config.get("turb_checkpoint_dir"),
            "turb_config_path": frontend_config.get("turb_config_path"),
            "global_stats_path": frontend_config.get("global_stats_path"),
            "device": runtime_device,
            "nca_steps_per_cycle": frontend_config.get("nca_steps_per_cycle", 16),
            "turb_nca_steps_per_macro": frontend_config.get("turb_nca_steps_per_macro", 8),
            "turb_nca_steps_per_cycle": frontend_config.get("turb_nca_steps_per_macro", 8),
            "max_iters": frontend_config.get("max_iters", 64),
            "mae_tol": frontend_config.get("mae_tol", 1e-4),
            "output_moments": frontend_config.get("output_moments", True),
            "output_phy_fields": frontend_config.get("output_phy_fields", True),
        }
        supported_parameters = inspect.signature(InferencePipeline.__init__).parameters
        pipeline_kwargs = {
            key: value
            for key, value in candidate_kwargs.items()
            if key in supported_parameters and value is not None
        }
        pipeline = InferencePipeline(**pipeline_kwargs)

    logger.info(
        "[Loader] inference pipeline ready | backend_root=%s | rans_checkpoint_dir=%s",
        backend_root,
        rans_checkpoint_dir,
    )

    return PipelineInfo(
        pipeline=pipeline,
        backend_root=str(backend_root),
        rans_checkpoint_dir=rans_checkpoint_dir,
        inference_config_path=str(inference_config_path) if inference_config_path else None,
    )
