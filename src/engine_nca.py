"""
engine_nca.py
NCA 推論引擎：管理狀態張量、驅動 model.forward()、渲染速度場。

放置於：NCA_workspace/03-gh-frontend/src/engine_nca.py
注意：ti.init() 由 main.py 統一呼叫，不在此重複。
"""

import base64
import logging

import cv2
import numpy as np
import taichi as ti
import torch

logger = logging.getLogger("RhinoBridge")


@ti.data_oriented
class NCAEngine:
    """
    NCA 推論引擎。
    職責：
      1. 接收 SDFEngine / ComputeEngine 已計算好的 mask/SDF numpy 陣列
      2. 組裝 NCA 初始 seed 狀態張量 [1, C, H, W]
      3. 每幀呼叫 step() 推進 model.forward()
      4. 將速度場渲染至 Taichi canvas 供 GUI 顯示
      5. 提供 get_jpeg_b64() 供 WebSocket 回傳給 Grasshopper
    """

    def __init__(
        self,
        model: torch.nn.Module,
        nca_channels: int,
        static_channels: int,
        data_res: int = 256,
        gui_res: int = 512,
    ):
        self.model = model
        self.nca_channels = nca_channels
        self.static_channels = static_channels
        self.phy_start = static_channels          # Ch2
        self.phy_end   = static_channels + 9      # Ch11（D2Q9 固定 9 通道）
        self.data_res = data_res
        self.gui_res  = gui_res
        self.device   = next(model.parameters()).device

        # ── Taichi 顯示畫布（RGB）────────────────────────────────────────
        self.canvas = ti.Vector.field(3, dtype=ti.f32, shape=(gui_res, gui_res))

        # ── 速度場 numpy buffer（data_res 解析度，RGB float32）────────────
        # PyTorch → numpy → Taichi 的橋接，避免 CUDA tensor 直接交換
        self._vel_np = np.zeros((data_res, data_res, 3), dtype=np.float32)
        self._vel_field = ti.Vector.field(3, dtype=ti.f32, shape=(data_res, data_res))

        # ── NCA 狀態張量 [1, C, H, W] ────────────────────────────────────
        self.state: torch.Tensor | None = None
        self._ready = False          # 幾何是否已初始化

        logger.info(
            f"[NCAEngine] 初始化 | res={data_res} | "
            f"ch={nca_channels} | static={static_channels} | device={self.device}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # 公開 API
    # ──────────────────────────────────────────────────────────────────────

    def reset_state(self, mask_np: np.ndarray, sdf_np: np.ndarray) -> None:
        """
        幾何改變後重置 NCA 狀態。
        由 ws_handler 在幾何更新且模式為 "nca" 時呼叫。

        Args:
            mask_np : [H, W] float32，來自 ComputeEngine.mask_field.to_numpy()
                      1.0 = 障礙物（固體），0.0 = 流體
            sdf_np  : [H, W] float32，來自 SDFEngine.sdf_field.to_numpy()
                      負值 = 固體內部，正值 = 流體
        """
        H, W = mask_np.shape
        C = self.nca_channels

        state = torch.zeros((1, C, H, W), dtype=torch.float32, device=self.device)

        # Ch 0: obstacle mask（直接複製）
        state[0, 0] = torch.from_numpy(mask_np).to(self.device)

        # Ch 1: SDF（縮放至 [-1, 1] 防止初期數值過大）
        sdf_norm = np.clip(sdf_np / (max(H, W) * 0.5), -1.0, 1.0).astype(np.float32)
        state[0, 1] = torch.from_numpy(sdf_norm).to(self.device)

        # Ch 2 (rho): 流體區域初始化為 1.0（LBM 標準密度）
        fluid = torch.from_numpy(1.0 - mask_np).to(self.device)
        state[0, self.phy_start] = fluid   # rho channel

        # 其餘物理 & 隱藏通道維持 0（靠模型自然演化）

        self.state = state
        self._ready = True
        logger.info(f"[NCAEngine] Seed 已重置，H={H} W={W}")

    @torch.no_grad()
    def step(self, n_steps: int = 1) -> None:
        """
        推進 NCA 演化 n_steps 步，並更新 Taichi 畫布。
        由 render_loop 每幀呼叫（通常 n_steps=1~4）。
        """
        if not self._ready or self.state is None:
            return

        x = self.state
        for _ in range(n_steps):
            x = self.model(x, update_rate=1.0)  # 推論時 update_rate=1.0（不 dropout）

            if torch.isnan(x).any():
                logger.warning("[NCAEngine] 偵測到 NaN！清除物理通道並繼續。")
                x[0, self.phy_start:] = 0.0
                break

        self.state = x
        self._render_velocity()

    def get_jpeg_b64(self, quality: int = 75) -> str:
        """
        將速度場畫布編碼為 JPEG base64 字串，供 WebSocket JSON 回傳給 GH。

        GH 端收到後可用 System.Drawing.Bitmap 解碼並貼到 Mesh 上顯示。
        """
        rgb_u8 = (self._vel_np * 255).clip(0, 255).astype(np.uint8)
        bgr = rgb_u8[..., ::-1]
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            return base64.b64encode(buf.tobytes()).decode("ascii")
        return ""

    def get_velocity_stats(self) -> dict:
        """回傳速度場統計值，供 GH 數值顯示用。"""
        if self.state is None:
            return {"ux_max": 0.0, "uy_max": 0.0, "speed_max": 0.0}

        s = self.state[0]
        rho = s[self.phy_start].cpu().numpy()
        jx  = s[self.phy_start + 3].cpu().numpy()   # Ch5 = jx (m3)
        jy  = s[self.phy_start + 5].cpu().numpy()   # Ch7 = jy (m5)
        safe_rho = np.clip(rho, 1e-2, None)
        ux = jx / safe_rho
        uy = jy / safe_rho
        speed = np.sqrt(ux**2 + uy**2)
        return {
            "ux_max":    float(np.abs(ux).max()),
            "uy_max":    float(np.abs(uy).max()),
            "speed_max": float(speed.max()),
            "speed_mean": float(speed.mean()),
        }

    # ──────────────────────────────────────────────────────────────────────
    # 內部：速度場渲染
    # ──────────────────────────────────────────────────────────────────────

    def _render_velocity(self) -> None:
        """把速度場轉成 Jet 熱力圖，寫入 _vel_np 與 Taichi canvas。"""
        s = self.state[0]

        # 取出物理通道
        rho = s[self.phy_start].cpu().numpy()                # Ch2: rho
        jx  = s[self.phy_start + 3].cpu().numpy()           # Ch5: jx
        jy  = s[self.phy_start + 5].cpu().numpy()           # Ch7: jy

        safe_rho = np.clip(rho, 1e-2, None)
        ux    = np.clip(jx / safe_rho, -5.0, 5.0)
        uy    = np.clip(jy / safe_rho, -5.0, 5.0)
        speed = np.sqrt(ux**2 + uy**2)

        # 歸一化速度大小 → Jet 顏色映射
        s_min, s_max = speed.min(), speed.max()
        speed_norm = ((speed - s_min) / (s_max - s_min + 1e-8) * 255).astype(np.uint8)
        jet_bgr = cv2.applyColorMap(speed_norm, cv2.COLORMAP_JET)  # [H,W,3] BGR
        jet_rgb = jet_bgr[..., ::-1].astype(np.float32) / 255.0    # → RGB [0,1]

        # 固體遮罩設為深灰
        obstacle = s[0].cpu().numpy() > 0.5
        jet_rgb[obstacle] = [0.15, 0.15, 0.15]

        self._vel_np[:] = jet_rgb

        # 傳到 Taichi 並 upsample 到 GUI 解析度
        self._vel_field.from_numpy(jet_rgb)
        self._upsample_canvas()

    @ti.kernel
    def _upsample_canvas(self):
        """[GPU] 最近鄰升採樣到 GUI 解析度。"""
        for i, j in self.canvas:
            si = i * self.data_res // self.gui_res
            sj = j * self.data_res // self.gui_res
            self.canvas[i, j] = self._vel_field[si, sj]
