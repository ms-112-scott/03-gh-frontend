"""
engine_nca.py  (視覺化升級版)
NCA 推論 + 多模式物理場後處理 + Taichi 渲染。

新增：
  - viz_mode 屬性（即時切換，無需重建引擎）
  - 蒲氏風力級數模式
  - lbm_to_ms 換算係數設定
  - get_beaufort_legend() 供 GH 顯示色卡

放置於：NCA_workspace/03-gh-frontend/src/engine_nca.py
"""

import base64
import logging

import cv2_test
import numpy as np
import taichi as ti
import torch

from physics_colorizer import colorize, VIZ_MODES, get_beaufort_legend

logger = logging.getLogger("RhinoBridge")


@ti.data_oriented
class NCAEngine:
    """
    NCA 推論引擎（含多模式物理場視覺化）。

    viz_mode 可以在運行中透過 WebSocket 訊息動態切換：
        "velocity"  - 速度量值 (plasma)
        "vorticity" - 渦度場
        "pressure"  - 壓力偏差
        "beaufort"  - 蒲氏風力級數（離散色階）
        "stress"    - 剪切應力 pxy
    """

    def __init__(
        self,
        model: torch.nn.Module,
        nca_channels: int,
        static_channels: int,
        data_res: int = 256,
        gui_res: int = 512,
        lbm_to_ms: float = 10.0,
        viz_mode: str = "velocity",
    ):
        self.model = model
        self.nca_channels = nca_channels
        self.static_channels = static_channels
        self.phy_start = static_channels  # Ch2
        self.data_res = data_res
        self.gui_res = gui_res
        self.device = next(model.parameters()).device

        # 視覺化設定
        self.lbm_to_ms = lbm_to_ms
        self._viz_mode = viz_mode if viz_mode in VIZ_MODES else "velocity"

        # ── Taichi 顯示畫布（RGB）──────────────────────────────────────
        self.canvas = ti.Vector.field(3, dtype=ti.f32, shape=(gui_res, gui_res))

        # ── 中間 numpy buffer（低解析度 RGB）────────────────────────────
        self._rgb_np = np.zeros((data_res, data_res, 3), dtype=np.float32)
        self._rgb_field = ti.Vector.field(3, dtype=ti.f32, shape=(data_res, data_res))

        # ── NCA 狀態張量 [1, C, H, W] ─────────────────────────────────
        self.state: torch.Tensor | None = None
        self._ready = False

        logger.info(
            f"[NCAEngine] 初始化 | res={data_res} | ch={nca_channels} | "
            f"static={static_channels} | lbm_to_ms={lbm_to_ms} | "
            f"viz={self._viz_mode} | device={self.device}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # 屬性
    # ──────────────────────────────────────────────────────────────────────

    @property
    def viz_mode(self) -> str:
        return self._viz_mode

    @viz_mode.setter
    def viz_mode(self, mode: str):
        if mode not in VIZ_MODES:
            raise ValueError(f"未知的視覺化模式 '{mode}'，可用: {VIZ_MODES}")
        if mode != self._viz_mode:
            self._viz_mode = mode
            logger.info(f"[NCAEngine] 切換視覺化模式 → {mode}")
            # 立即重新渲染（若狀態存在）
            if self._ready and self.state is not None:
                self._render()

    # ──────────────────────────────────────────────────────────────────────
    # 公開 API
    # ──────────────────────────────────────────────────────────────────────

    def reset_state(self, mask_np: np.ndarray, sdf_np: np.ndarray) -> None:
        """
        幾何改變後重置 NCA 狀態。
        由 ws_handler 在幾何更新且模式為 "nca" 時呼叫。

        Args:
            mask_np : [H, W] float32 (1=障礙物 0=流體)  → from engine_mask.mask_field
            sdf_np  : [H, W] float32 (負=固體內 正=流體) → from engine_sdf.sdf_field
        """
        H, W = mask_np.shape
        C = self.nca_channels

        state = torch.zeros((1, C, H, W), dtype=torch.float32, device=self.device)

        # Ch0: obstacle mask
        state[0, 0] = torch.from_numpy(mask_np).to(self.device)
        # Ch1: SDF（縮放至 [-1,1]）
        sdf_norm = np.clip(sdf_np / (max(H, W) * 0.5), -1.0, 1.0).astype(np.float32)
        state[0, 1] = torch.from_numpy(sdf_norm).to(self.device)
        # Ch2 (rho): 流體區域初始化為 1.0
        fluid = torch.from_numpy(1.0 - mask_np).to(self.device)
        state[0, self.phy_start] = fluid

        self.state = state
        self._ready = True
        logger.info(f"[NCAEngine] Seed 已重置 H={H} W={W}")

    @torch.no_grad()
    def step(self, n_steps: int = 1) -> None:
        """
        推進 NCA 演化 n_steps 步，然後依目前 viz_mode 更新畫布。
        由 render_loop 每幀呼叫。
        """
        if not self._ready or self.state is None:
            return

        x = self.state
        for _ in range(n_steps):
            x = self.model(x, update_rate=1.0)
            if torch.isnan(x).any():
                logger.warning("[NCAEngine] 偵測到 NaN，清除物理通道。")
                x[0, self.phy_start :] = 0.0
                break

        self.state = x
        self._render()

    def set_viz_mode(self, mode: str) -> None:
        """由 WebSocket 訊息呼叫的入口（等同 self.viz_mode = mode）。"""
        self.viz_mode = mode

    def get_jpeg_b64(self, quality: int = 75) -> str:
        """編碼為 JPEG base64，供 WebSocket 回傳給 GH。"""
        rgb_u8 = (self._rgb_np * 255).clip(0, 255).astype(np.uint8)
        bgr = rgb_u8[..., ::-1]
        ok, buf = cv2_test.imencode(
            ".jpg", bgr, [cv2_test.IMWRITE_JPEG_QUALITY, quality]
        )
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

    def get_velocity_stats(self) -> dict:
        """回傳速度場統計，供 GH 數值顯示。"""
        if self.state is None:
            return {}
        s = self.state[0].cpu().numpy()
        rho = s[self.phy_start]
        jx = s[self.phy_start + 3]
        jy = s[self.phy_start + 5]
        safe_rho = np.clip(rho, 1e-2, None)
        ux = jx / safe_rho
        uy = jy / safe_rho
        speed_lbm = np.sqrt(ux**2 + uy**2)
        speed_ms = speed_lbm * self.lbm_to_ms
        fluid = s[0] < 0.5
        return {
            "viz_mode": self._viz_mode,
            "speed_max_ms": float(speed_ms[fluid].max()) if fluid.any() else 0.0,
            "speed_mean_ms": float(speed_ms[fluid].mean()) if fluid.any() else 0.0,
            "beaufort_max": (
                int(_ms_to_beaufort(float(speed_ms[fluid].max()))) if fluid.any() else 0
            ),
        }

    @staticmethod
    def available_viz_modes() -> list[str]:
        return VIZ_MODES

    @staticmethod
    def beaufort_legend() -> list[dict]:
        """回傳蒲氏級數圖例，供 GH 顯示色卡說明。"""
        return get_beaufort_legend()

    # ──────────────────────────────────────────────────────────────────────
    # 內部：物理場渲染
    # ──────────────────────────────────────────────────────────────────────

    def _render(self) -> None:
        """把 state[:11] 透過 physics_colorizer 轉為 RGB 並推到 Taichi。"""
        state_np = self.state[0, :11].cpu().numpy()  # [11, H, W]

        try:
            rgb = colorize(
                state_np,
                mode=self._viz_mode,
                lbm_to_ms=self.lbm_to_ms,
            )  # [H, W, 3] float32
        except Exception as e:
            logger.error(f"[NCAEngine] colorize 失敗: {e}")
            return

        self._rgb_np[:] = rgb
        self._rgb_field.from_numpy(rgb)
        self._upsample_canvas()

    @ti.kernel
    def _upsample_canvas(self):
        """[GPU] 最近鄰升採樣到 GUI 解析度。"""
        for i, j in self.canvas:
            si = i * self.data_res // self.gui_res
            sj = j * self.data_res // self.gui_res
            self.canvas[i, j] = self._rgb_field[si, sj]


# ──────────────────────────────────────────────────────────────────────────
# 輔助函數
# ──────────────────────────────────────────────────────────────────────────


def _ms_to_beaufort(speed_ms: float) -> int:
    """將 m/s 速度轉換為蒲氏風力級數（0-12）。"""
    from physics_colorizer import BEAUFORT_LIMITS_MS

    for level, upper in enumerate(BEAUFORT_LIMITS_MS):
        if speed_ms < upper:
            return level
    return 12
