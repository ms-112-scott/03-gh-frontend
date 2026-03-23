from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np
import taichi as ti
import torch
import torch.nn.functional as F

from src.visualization.physics_colorizer import (
    BEAUFORT_LIMITS_MS,
    VIZ_MODES,
    colorize,
    get_beaufort_legend,
)


logger = logging.getLogger("RhinoBridge")

CONDITION_KEYS = (
    "Re",
    "U_inlet",
    "L_char",
    "BCR",
    "AR",
    "perimeter_density",
    "U_peak_ratio",
)

CONDITION_ALIASES = {
    "perim_density": "perimeter_density",
}

DEFAULT_CONDITIONS = {
    "Re": 1.5e7,
    "U_inlet": 5.0,
    "L_char": 50.0,
    "BCR": 0.12,
    "AR": 2.0,
    "perimeter_density": 0.015,
    "U_peak_ratio": 0.01,
}


@dataclass
class PhysicalConditions:
    Re: float = DEFAULT_CONDITIONS["Re"]
    U_inlet: float = DEFAULT_CONDITIONS["U_inlet"]
    L_char: float = DEFAULT_CONDITIONS["L_char"]
    BCR: float = DEFAULT_CONDITIONS["BCR"]
    AR: float = DEFAULT_CONDITIONS["AR"]
    perimeter_density: float = DEFAULT_CONDITIONS["perimeter_density"]
    U_peak_ratio: float = DEFAULT_CONDITIONS["U_peak_ratio"]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "PhysicalConditions":
        if not data:
            return cls()

        normalized: dict[str, Any] = {}
        for key, value in data.items():
            canonical = CONDITION_ALIASES.get(key, key)
            normalized[canonical] = value

        kwargs = {key: float(normalized.get(key, DEFAULT_CONDITIONS[key])) for key in CONDITION_KEYS}
        return cls(**kwargs)

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        values = [[getattr(self, key) for key in CONDITION_KEYS]]
        return torch.tensor(values, dtype=torch.float32, device=device)

    def to_dict(self) -> dict[str, float]:
        return {key: float(getattr(self, key)) for key in CONDITION_KEYS}


@dataclass
class GeometryPayload:
    mask: np.ndarray
    sdf: np.ndarray


@ti.data_oriented
class NCAEngine:
    """
    Frontend inference adapter.

    This class does not perform ML core computation itself. It only:
    - packs frontend geometry/condition data for the backend pipeline,
    - stores latest physical-unit outputs from the backend,
    - derives lightweight frontend stats,
    - renders visualization RGB for Taichi display.
    """

    def __init__(
        self,
        pipeline: Any,
        data_res: int = 256,
        gui_res: int = 512,
        viz_mode: str = "velocity",
    ):
        self.pipeline = pipeline
        self.data_res = data_res
        self.gui_res = gui_res
        self.device = getattr(pipeline, "_device", torch.device("cpu"))
        self._viz_mode = viz_mode if viz_mode in VIZ_MODES else "velocity"

        self.canvas = ti.Vector.field(3, dtype=ti.f32, shape=(gui_res, gui_res))
        self._rgb_np = np.zeros((data_res, data_res, 3), dtype=np.float32)
        self._rgb_field = ti.Vector.field(3, dtype=ti.f32, shape=(data_res, data_res))

        self._geometry: GeometryPayload | None = None
        self._conditions = PhysicalConditions()
        self._latest_output: dict[str, Any] | None = None
        self._ready = False

    @property
    def viz_mode(self) -> str:
        return self._viz_mode

    @viz_mode.setter
    def viz_mode(self, mode: str) -> None:
        if mode not in VIZ_MODES:
            raise ValueError(f"Unsupported viz_mode '{mode}'. Available: {VIZ_MODES}")
        self._viz_mode = mode
        if self._latest_output is not None:
            self._render_latest()

    def set_viz_mode(self, mode: str) -> None:
        self.viz_mode = mode

    @staticmethod
    def available_viz_modes() -> list[str]:
        return VIZ_MODES

    @staticmethod
    def beaufort_legend() -> list[dict]:
        return get_beaufort_legend()

    @property
    def conditions(self) -> dict[str, float]:
        return self._conditions.to_dict()

    @property
    def latest_output(self) -> dict[str, Any] | None:
        return self._latest_output

    def set_geometry(
        self,
        mask_np: np.ndarray,
        sdf_np: np.ndarray,
        *,
        conditions: Mapping[str, Any] | PhysicalConditions | None = None,
        cold: bool = True,
    ) -> None:
        self._geometry = GeometryPayload(
            mask=np.asarray(mask_np, dtype=np.float32),
            sdf=np.asarray(sdf_np, dtype=np.float32),
        )
        if conditions is not None:
            self._conditions = (
                conditions
                if isinstance(conditions, PhysicalConditions)
                else PhysicalConditions.from_mapping(conditions)
            )
        self._reset_pipeline(cold=cold)

    def update_conditions(self, conditions: Mapping[str, Any] | PhysicalConditions) -> None:
        self._conditions = (
            conditions
            if isinstance(conditions, PhysicalConditions)
            else PhysicalConditions.from_mapping(conditions)
        )
        if self._geometry is None:
            return

        if not self._ready:
            self._reset_pipeline(cold=True)
            return

        self.pipeline.update_conditions(self._conditions.to_tensor(self.device))
        self._latest_output = None

    def restart(self, cold: bool = False) -> None:
        if not self._ready:
            return
        self.pipeline.restart(cold=cold)
        self._latest_output = None

    def toggle_module(self, name: str, enabled: bool) -> None:
        self.pipeline.toggle_module(name, enabled)

    def set_output_toggles(
        self,
        *,
        output_moments: bool | None = None,
        output_phy_fields: bool | None = None,
    ) -> None:
        if output_moments is not None:
            self.pipeline.rans_engine.output_moments = output_moments
        if output_phy_fields is not None:
            self.pipeline.rans_engine.output_phy_fields = output_phy_fields

    def step(self, n_steps: int = 1) -> None:
        if not self._ready:
            return
        self._latest_output = self.pipeline.step_n(n_steps)
        self._render_latest()

    def get_result(self) -> dict[str, Any]:
        if not self._ready:
            return {"moments": None, "phy_fields": None, "meta": {}}
        self._latest_output = self.pipeline.get_result()
        self._render_latest()
        return self._latest_output

    def get_jpeg_b64(self, quality: int = 75) -> str:
        rgb_u8 = (self._rgb_np * 255).clip(0, 255).astype(np.uint8)
        bgr = rgb_u8[..., ::-1]
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

    def get_velocity_stats(self) -> dict[str, Any]:
        if not self._latest_output:
            return {"conditions": self.conditions}

        phy_fields = self._latest_output.get("phy_fields")
        if phy_fields is None:
            return {"conditions": self.conditions}

        fields_np = self._to_numpy(phy_fields)
        speed_ms = fields_np[2]
        mask = self._geometry.mask if self._geometry is not None else np.zeros_like(speed_ms)
        fluid = mask < 0.5

        speed_max = float(speed_ms[fluid].max()) if fluid.any() else 0.0
        speed_mean = float(speed_ms[fluid].mean()) if fluid.any() else 0.0

        return {
            "model_type": "inference_pipeline",
            "viz_mode": self._viz_mode,
            "speed_max_ms": speed_max,
            "speed_mean_ms": speed_mean,
            "beaufort_max": int(_ms_to_beaufort(speed_max)) if fluid.any() else 0,
            "conditions": self.conditions,
            "meta": self._latest_output.get("meta", {}),
        }

    def _reset_pipeline(self, *, cold: bool) -> None:
        if self._geometry is None:
            return

        static_2ch = self._build_static_tensor(self._geometry.mask, self._geometry.sdf)
        global_static_2ch = self._build_global_tensor(static_2ch)
        conditions_7d = self._conditions.to_tensor(self.device)

        self.pipeline.reset(
            static_2ch=static_2ch,
            global_static_2ch=global_static_2ch,
            conditions_7d=conditions_7d,
            cold=cold,
        )
        self._ready = True
        self._latest_output = None

    def _build_static_tensor(self, mask_np: np.ndarray, sdf_np: np.ndarray) -> torch.Tensor:
        mask_t = torch.from_numpy(mask_np.astype(np.float32)).to(self.device)
        sdf_t = torch.from_numpy(sdf_np.astype(np.float32)).to(self.device)
        return torch.stack([mask_t, sdf_t], dim=0).unsqueeze(0)

    def _build_global_tensor(self, static_2ch: torch.Tensor) -> torch.Tensor:
        if static_2ch.shape[-2:] == (256, 256):
            return static_2ch
        return F.interpolate(static_2ch, size=(256, 256), mode="bilinear", align_corners=False)

    def _render_latest(self) -> None:
        if not self._latest_output or self._geometry is None:
            return

        moments = self._latest_output.get("moments")
        if moments is None:
            return

        moments_np = self._to_numpy(moments)
        viz_state = np.concatenate(
            [
                self._geometry.mask[None, ...],
                self._geometry.sdf[None, ...],
                moments_np,
            ],
            axis=0,
        ).astype(np.float32)

        rgb = colorize(viz_state, mode=self._viz_mode)
        self._rgb_np[:] = rgb
        self._rgb_field.from_numpy(rgb)
        self._upsample_canvas()

    def _to_numpy(self, tensor_or_array: Any) -> np.ndarray:
        if isinstance(tensor_or_array, np.ndarray):
            return tensor_or_array
        if torch.is_tensor(tensor_or_array):
            return tensor_or_array.detach().cpu().numpy()
        raise TypeError(f"Unsupported array type: {type(tensor_or_array)!r}")

    @ti.kernel
    def _upsample_canvas(self) -> None:
        for i, j in self.canvas:
            si = i * self.data_res // self.gui_res
            sj = j * self.data_res // self.gui_res
            self.canvas[i, j] = self._rgb_field[si, sj]


def _ms_to_beaufort(speed_ms: float) -> int:
    for level, upper in enumerate(BEAUFORT_LIMITS_MS):
        if speed_ms < upper:
            return level
    return 12
