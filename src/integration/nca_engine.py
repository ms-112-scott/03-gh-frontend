"""
engine_nca.py  (VAENCA 整合版)
================================
NCA 推論引擎，同時支援：

  VAENCA (新)
    step() 呼叫 model.rollout(x, target_phy, n_steps)
    target_phy = 當前物理通道（self-conditioning）
    NCA 權重由 VAE encoder-decoder 決定，一次計算全部 n_steps

  LBMInspiredNCA (舊，向下相容)
    step() 迴圈呼叫 model(x, update_rate=1.0)

視覺化功能不變：
  velocity / vorticity / pressure / beaufort / stress
  Taichi GPU 升採樣 → gui.canvas
  JPEG base64 → WebSocket → Grasshopper
"""

import base64
import logging
import numpy as np
import taichi as ti
import torch

import cv2
from src.visualization.physics_colorizer import (
    VIZ_MODES,
    colorize,
    get_beaufort_legend,
)

logger = logging.getLogger("RhinoBridge")


# ─────────────────────────────────────────────────────────────────────────────

@ti.data_oriented
class NCAEngine:
    """
    NCA 推論引擎。

    Args:
        model           : nn.Module（VAENCA 或 LBMInspiredNCA，eval mode）
        model_type      : "vaenca" | "lbm_inspired"
        total_channels  : NCA 狀態總通道數
        static_channels : 靜態通道數（mask + SDF = 2）
        phy_channels    : 物理矩通道數（D2Q9 = 9）
        data_res        : 計算解析度（例 256）
        gui_res         : Taichi GUI 解析度（例 512）
        lbm_to_ms       : LBM 速度單位 → m/s 換算係數
        viz_mode        : 初始視覺化模式
    """

    def __init__(
        self,
        model:           torch.nn.Module,
        model_type:      str   = "vaenca",
        total_channels:  int   = 32,
        static_channels: int   = 2,
        phy_channels:    int   = 9,
        data_res:        int   = 256,
        gui_res:         int   = 512,
        lbm_to_ms:       float = 10.0,
        viz_mode:        str   = "velocity",
    ):
        self.model           = model
        self.model_type      = model_type
        self.total_channels  = total_channels
        self.static_ch       = static_channels
        self.phy_ch          = phy_channels
        self.phy_start       = static_channels      # alias（向下相容）
        self.data_res        = data_res
        self.gui_res         = gui_res
        self.device          = next(model.parameters()).device
        self.lbm_to_ms       = lbm_to_ms
        self._viz_mode       = viz_mode if viz_mode in VIZ_MODES else "velocity"

        # ── Taichi RGB 畫布（GUI 解析度）────────────────────────────────
        self.canvas    = ti.Vector.field(3, dtype=ti.f32, shape=(gui_res, gui_res))

        # ── numpy 中繼 buffer + Taichi field（計算解析度）───────────────
        self._rgb_np    = np.zeros((data_res, data_res, 3), dtype=np.float32)
        self._rgb_field = ti.Vector.field(3, dtype=ti.f32, shape=(data_res, data_res))

        # ── NCA 狀態 [1, C, H, W] ───────────────────────────────────────
        self.state: torch.Tensor | None = None
        self._ready = False

        logger.info(
            f"[NCAEngine] init | type={model_type} | res={data_res} "
            f"| total_ch={total_channels} | static={static_channels} "
            f"| phy={phy_channels} | lbm_to_ms={lbm_to_ms} "
            f"| viz={self._viz_mode} | device={self.device}"
        )

    # ── viz_mode 屬性 ────────────────────────────────────────────────────────

    @property
    def viz_mode(self) -> str:
        return self._viz_mode

    @viz_mode.setter
    def viz_mode(self, mode: str):
        if mode not in VIZ_MODES:
            raise ValueError(f"未知 viz_mode '{mode}'，可用: {VIZ_MODES}")
        if mode != self._viz_mode:
            self._viz_mode = mode
            logger.info(f"[NCAEngine] viz_mode → {mode}")
            if self._ready and self.state is not None:
                self._render()

    def set_viz_mode(self, mode: str) -> None:
        """WebSocket 呼叫入口。"""
        self.viz_mode = mode

    # ── 公開 API ─────────────────────────────────────────────────────────────

    def reset_state(self, mask_np: np.ndarray, sdf_np: np.ndarray) -> None:
        """
        幾何改變後重置 NCA 狀態張量。

        Layout:
            [0]        : obstacle mask  (1=固體)
            [1]        : SDF norm       ([-1,1])
            [2]        : rho            (流體=1.0)
            [3..9]     : 其餘 moments   (=0)
            [11..31]   : hidden         (小隨機擾動)
        """
        H, W = mask_np.shape
        C    = self.total_channels

        state = torch.zeros((1, C, H, W), dtype=torch.float32, device=self.device)

        # Ch0: obstacle mask
        state[0, 0] = torch.from_numpy(mask_np.astype(np.float32)).to(self.device)

        # Ch1: SDF 縮放至 [-1, 1]
        sdf_norm = np.clip(
            sdf_np / (max(H, W) * 0.5), -1.0, 1.0
        ).astype(np.float32)
        state[0, 1] = torch.from_numpy(sdf_norm).to(self.device)

        # Ch[static_ch + 0] = rho：流體區域初始化為 1.0
        fluid_mask = torch.from_numpy((1.0 - mask_np).astype(np.float32)).to(self.device)
        state[0, self.static_ch] = fluid_mask

        # hidden channels：微小隨機擾動打破完全對稱
        s  = self.static_ch
        p  = self.phy_ch
        hc = C - s - p
        if hc > 0:
            state[0, s + p:] = (
                torch.randn(hc, H, W, device=self.device) * 0.01
            )

        self.state  = state
        self._ready = True
        logger.info(f"[NCAEngine] state reset | H={H} W={W} C={C}")

    @torch.no_grad()
    def step(self, n_steps: int = 1) -> None:
        """
        推進 NCA 演化 n_steps 步，更新畫布。

        VAENCA 路徑：
            一次 rollout(x, target_phy, n_steps)，效率高。
            target_phy = 當前物理通道（self-conditioning）：
            encoder 讀取當前流場，產生使狀態趨向一致解的 NCA 權重。

        LBMInspiredNCA 路徑（向下相容）：
            迴圈 n 次 model(x, update_rate=1.0)。
        """
        if not self._ready or self.state is None:
            return

        x = self.state

        if self.model_type == "vaenca":
            # ── VAENCA rollout ───────────────────────────────────────────
            s   = self.static_ch
            p   = self.phy_ch
            # self-condition：以當前物理矩通道為 target
            tgt = x[:, s : s + p].clone()          # [1, 9, H, W]
            x   = self.model.rollout(x, tgt, n_steps)

            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.warning("[NCAEngine] VAENCA NaN/Inf detected，復原物理通道。")
                x = self.state.clone()
                x[0, s : s + p] = 0.0

        else:
            # ── LBMInspiredNCA 逐步迴圈 ─────────────────────────────────
            for _ in range(n_steps):
                x = self.model(x, update_rate=1.0)
                if torch.isnan(x).any():
                    logger.warning("[NCAEngine] NaN detected，清除物理通道。")
                    x[0, self.phy_start:] = 0.0
                    break

        self.state = x
        self._render()

    def get_jpeg_b64(self, quality: int = 75) -> str:
        """當前畫面編碼為 JPEG base64，供 WebSocket 回傳。"""
        rgb_u8 = (self._rgb_np * 255).clip(0, 255).astype(np.uint8)
        bgr    = rgb_u8[..., ::-1]
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

    def get_velocity_stats(self) -> dict:
        """
        回傳速度場統計供 GH 顯示。
        VAENCA 額外回傳 receptor_weights（5 種受體的注意力比例）。
        """
        if self.state is None:
            return {}

        s_np     = self.state[0].cpu().numpy()
        s        = self.static_ch
        rho      = s_np[s]
        jx       = s_np[s + 3]
        jy       = s_np[s + 5]
        safe_rho = np.clip(rho, 1e-2, None)
        ux       = jx / safe_rho
        uy       = jy / safe_rho
        speed_lbm = np.sqrt(ux**2 + uy**2)
        speed_ms  = speed_lbm * self.lbm_to_ms
        fluid     = s_np[0] < 0.5          # mask < 0.5 → 流體

        stats: dict = {
            "model_type":    self.model_type,
            "viz_mode":      self._viz_mode,
            "speed_max_ms":  float(speed_ms[fluid].max())  if fluid.any() else 0.0,
            "speed_mean_ms": float(speed_ms[fluid].mean()) if fluid.any() else 0.0,
            "beaufort_max":  (
                int(_ms_to_beaufort(float(speed_ms[fluid].max()))) if fluid.any() else 0
            ),
        }

        # VAENCA 專屬：回傳感知受體權重
        if self.model_type == "vaenca":
            try:
                p   = self.phy_ch
                tgt = self.state[:, s : s + p]
                z   = self.model.encoder(tgt)
                _, _, _, w_perc = self.model.decoder(z)
                wp  = w_perc[0].cpu().tolist()      # [5]
                stats["receptor_weights"] = {
                    "A_topo":    round(wp[0], 3),
                    "B_stream":  round(wp[1], 3),
                    "C_mrt":     round(wp[2], 3),
                    "D_learned": round(wp[3], 3),
                    "E_hormone": round(wp[4], 3),
                }
            except Exception as e:
                logger.debug(f"[NCAEngine] receptor_weights 擷取失敗: {e}")

        return stats

    @staticmethod
    def available_viz_modes() -> list[str]:
        return VIZ_MODES

    @staticmethod
    def beaufort_legend() -> list[dict]:
        return get_beaufort_legend()

    # ── 內部渲染 ──────────────────────────────────────────────────────────────

    def _render(self) -> None:
        """
        把 state 前 (static_ch + phy_ch) 個通道（= 11）
        透過 physics_colorizer 轉為 RGB 並推到 Taichi canvas。
        """
        n_ch     = self.static_ch + self.phy_ch      # = 11
        state_np = self.state[0, :n_ch].cpu().numpy()  # [11, H, W]

        try:
            rgb = colorize(
                state_np,
                mode=self._viz_mode,
                lbm_to_ms=self.lbm_to_ms,
            )   # [H, W, 3] float32
        except Exception as e:
            logger.error(f"[NCAEngine] colorize 失敗: {e}")
            return

        self._rgb_np[:] = rgb
        self._rgb_field.from_numpy(rgb)
        self._upsample_canvas()

    @ti.kernel
    def _upsample_canvas(self):
        """[GPU] 最近鄰升採樣：計算解析度 → GUI 解析度。"""
        for i, j in self.canvas:
            si = i * self.data_res // self.gui_res
            sj = j * self.data_res // self.gui_res
            self.canvas[i, j] = self._rgb_field[si, sj]


# ─────────────────────────────────────────────────────────────────────────────
# 輔助
# ─────────────────────────────────────────────────────────────────────────────

def _ms_to_beaufort(speed_ms: float) -> int:
    from src.visualization.physics_colorizer import BEAUFORT_LIMITS_MS
    for level, upper in enumerate(BEAUFORT_LIMITS_MS):
        if speed_ms < upper:
            return level
    return 12
