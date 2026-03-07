"""
physics_colorizer.py
NCA 推論結果後處理：從 state[:11] 提取物理量並轉換為 RGB 影像。

放置於：NCA_workspace/03-gh-frontend/src/physics_colorizer.py

支援五種視覺化模式（可即時切換）：
  "velocity"  - 速度量值       (plasma)
  "vorticity" - 渦度場          (自訂黃橘黑綠青)
  "pressure"  - 壓力場          (RdBu_r)
  "beaufort"  - 蒲氏風力級數   (12 級離散色階)
  "stress"    - 剪切應力 pxy   (coolwarm)
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")   # 無頭後端，不開視窗
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap, BoundaryNorm
from typing import Optional
from numpy.typing import NDArray


# ══════════════════════════════════════════════════════════════════════════════
# 蒲氏風力級數定義
# ══════════════════════════════════════════════════════════════════════════════

# 各級上限速度 (m/s)，第 12 級無上限
BEAUFORT_LIMITS_MS: list[float] = [
    0.3,   # 0 → 1
    1.5,   # 1 → 2
    3.3,   # 2 → 3
    5.5,   # 3 → 4
    7.9,   # 4 → 5
    10.7,  # 5 → 6
    13.8,  # 6 → 7
    17.1,  # 7 → 8
    20.7,  # 8 → 9
    24.4,  # 9 → 10
    28.4,  # 10 → 11
    32.6,  # 11 → 12
    999.0, # 12 上限（颶風）
]

BEAUFORT_LABELS: list[str] = [
    "0 靜風", "1 軟風", "2 輕風", "3 微風",
    "4 和風", "5 清風", "6 強風", "7 疾風",
    "8 大風", "9 烈風", "10 狂風", "11 暴風", "12 颶風",
]

# 13 個邊界 = 12 個區間
BEAUFORT_BOUNDS: NDArray = np.array([0.0] + BEAUFORT_LIMITS_MS)

# 每級顏色（RGB 0-1），從冷色→暖色→紅→紫，讓建築師一眼看出危險程度
BEAUFORT_COLORS: list[tuple[float, float, float]] = [
    (0.90, 0.97, 1.00),  # 0  靜風   淡水藍
    (0.68, 0.88, 0.98),  # 1  軟風   天藍
    (0.40, 0.76, 0.96),  # 2  輕風   藍
    (0.20, 0.63, 0.86),  # 3  微風   中藍
    (0.16, 0.84, 0.66),  # 4  和風   青綠
    (0.31, 0.93, 0.31),  # 5  清風   翠綠
    (0.88, 0.93, 0.13),  # 6  強風   黃
    (1.00, 0.75, 0.02),  # 7  疾風   橙黃
    (1.00, 0.48, 0.00),  # 8  大風   橙
    (0.96, 0.22, 0.06),  # 9  烈風   紅橙
    (0.80, 0.04, 0.04),  # 10 狂風   深紅
    (0.58, 0.00, 0.58),  # 11 暴風   紫
    (0.25, 0.00, 0.35),  # 12 颶風   深紫
]

def _build_beaufort_cmap() -> tuple[LinearSegmentedColormap, BoundaryNorm]:
    """建立 13 色離散 Colormap 與對應的 BoundaryNorm。"""
    cmap = LinearSegmentedColormap.from_list(
        "beaufort", BEAUFORT_COLORS, N=13
    )
    cmap.set_bad(color=(0.4, 0.4, 0.4))    # NaN → 深灰（固體）
    norm = BoundaryNorm(BEAUFORT_BOUNDS, ncolors=13)
    return cmap, norm


def _build_vorticity_cmap() -> LinearSegmentedColormap:
    """渦度：負渦（順時針）→黃橘，中心黑，正渦（逆時針）→綠青"""
    colors = [
        (1.00, 1.00, 0.00),   # 強負渦  黃
        (0.95, 0.49, 0.02),   # 負渦    橙
        (0.00, 0.00, 0.00),   # 零      黑
        (0.18, 0.98, 0.53),   # 正渦    翠綠
        (0.00, 1.00, 1.00),   # 強正渦  青
    ]
    cmap = LinearSegmentedColormap.from_list("vorticity_cmap", colors)
    cmap.set_bad(color=(0.5, 0.5, 0.5))
    return cmap


# ══════════════════════════════════════════════════════════════════════════════
# 內部通用映射工具
# ══════════════════════════════════════════════════════════════════════════════

def _apply_colormap(
    data: NDArray[np.float32],
    cmap: matplotlib.colors.Colormap,
    norm: matplotlib.colors.Normalize,
    mask: Optional[NDArray] = None,
    obstacle_gray: float = 0.35,
) -> NDArray[np.float32]:
    """
    通用：把 [H, W] float 陣列映射為 [H, W, 3] float32 RGB。
    mask: 1=固體，0=流體。固體像素統一設為 obstacle_gray 深灰。
    """
    plot_data = data.copy()
    if mask is not None:
        plot_data[mask > 0.5] = np.nan     # 讓 cmap.set_bad 處理

    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba = mapper.to_rgba(plot_data)       # [H, W, 4] float64
    rgb  = rgba[:, :, :3].astype(np.float32)

    if mask is not None:
        rgb[mask > 0.5] = obstacle_gray    # 固體強制覆蓋為深灰（銳利邊界）

    return rgb


# ══════════════════════════════════════════════════════════════════════════════
# 物理量提取（從 NCA state [1, 11, H, W] 或 [11, H, W]）
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsFields:
    """從 NCA state 提取並快取所有衍生物理量，避免重複計算。"""

    # Channel layout (訓練時約定)
    CH_MASK = 0;  CH_SDF = 1
    CH_RHO  = 2;  CH_E   = 3;  CH_EPS = 4
    CH_JX   = 5;  CH_QX  = 6
    CH_JY   = 7;  CH_QY  = 8
    CH_PXX  = 9;  CH_PXY = 10

    def __init__(
        self,
        state_np: NDArray[np.float32],
        lbm_to_ms: float = 10.0,
    ):
        """
        Args:
            state_np  : [11, H, W] 或 [1, 11, H, W] 的 numpy 陣列
            lbm_to_ms : LBM 單位速度 → m/s 的換算係數
                        （例：lbm u=1.0 對應 lbm_to_ms m/s）
                        根據實際模型和風洞縮尺比設定。
        """
        if state_np.ndim == 4:
            state_np = state_np[0]              # [11, H, W]
        assert state_np.shape[0] >= 11, "至少需要 11 個通道"

        self.lbm_to_ms = lbm_to_ms
        s = state_np                            # alias

        # ── 靜態場 ──────────────────────────────────────────────────────
        self.mask: NDArray = (s[self.CH_MASK] > 0.5).astype(np.float32)  # [H,W] 0/1
        self.sdf:  NDArray = s[self.CH_SDF]

        # ── LBM Moments ─────────────────────────────────────────────────
        self.rho: NDArray = s[self.CH_RHO]
        self.jx:  NDArray = s[self.CH_JX]
        self.jy:  NDArray = s[self.CH_JY]
        self.pxx: NDArray = s[self.CH_PXX]
        self.pxy: NDArray = s[self.CH_PXY]

        # ── 衍生量（延遲計算） ──────────────────────────────────────────
        self._ux: NDArray | None = None
        self._uy: NDArray | None = None
        self._speed_lbm:  NDArray | None = None
        self._speed_ms:   NDArray | None = None
        self._vorticity:  NDArray | None = None
        self._pressure:   NDArray | None = None

    @property
    def ux(self) -> NDArray:
        if self._ux is None:
            safe_rho = np.clip(self.rho, 1e-2, None)
            self._ux = np.clip(self.jx / safe_rho, -5.0, 5.0)
        return self._ux

    @property
    def uy(self) -> NDArray:
        if self._uy is None:
            safe_rho = np.clip(self.rho, 1e-2, None)
            self._uy = np.clip(self.jy / safe_rho, -5.0, 5.0)
        return self._uy

    @property
    def speed_lbm(self) -> NDArray:
        """速度量值（LBM 單位）"""
        if self._speed_lbm is None:
            self._speed_lbm = np.sqrt(self.ux**2 + self.uy**2)
        return self._speed_lbm

    @property
    def speed_ms(self) -> NDArray:
        """速度量值（m/s，乘以換算係數）"""
        if self._speed_ms is None:
            self._speed_ms = self.speed_lbm * self.lbm_to_ms
        return self._speed_ms

    @property
    def vorticity(self) -> NDArray:
        """
        Z 軸渦度 ω = ∂uy/∂x - ∂ux/∂y
        使用 numpy.gradient（中央差分），固體內部渦度設 0。
        """
        if self._vorticity is None:
            duy_dx = np.gradient(self.uy, axis=1)
            dux_dy = np.gradient(self.ux, axis=0)
            w = duy_dx - dux_dy
            w[self.mask > 0.5] = 0.0           # 固體無渦度
            self._vorticity = w
        return self._vorticity

    @property
    def pressure(self) -> NDArray:
        """
        壓力 p = rho * cs²，其中 cs² = 1/3（D2Q9 標準值）。
        返回相對於流場平均值的偏差（突出高低壓區）。
        """
        if self._pressure is None:
            p = self.rho / 3.0
            fluid_mask = 1.0 - self.mask
            fluid_mean = (p * fluid_mask).sum() / (fluid_mask.sum() + 1e-8)
            self._pressure = (p - fluid_mean) * fluid_mask
        return self._pressure

    def beaufort_scale(self) -> NDArray[np.int8]:
        """每個像素的蒲氏風力級數 (0-12)。"""
        spd = self.speed_ms
        scale = np.zeros_like(spd, dtype=np.int8)
        for level, upper in enumerate(BEAUFORT_LIMITS_MS[:-1]):
            scale[spd >= upper] = level + 1
        scale[self.mask > 0.5] = -1   # 固體設 -1，渲染時特別處理
        return scale


# ══════════════════════════════════════════════════════════════════════════════
# 公開彩色化函數
# ══════════════════════════════════════════════════════════════════════════════

def colorize_velocity(
    fields: PhysicsFields,
    u_max: Optional[float] = None,
    cmap_name: str = "plasma",
) -> NDArray[np.float32]:
    """
    速度量值熱力圖。
    u_max: LBM 單位的上限速度（None → 自動取場最大值）。
    """
    spd  = fields.speed_lbm
    vmax = u_max if u_max is not None else float(np.percentile(spd[fields.mask < 0.5], 98))
    vmax = max(vmax, 1e-6)
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    cmap.set_bad(color=(0.35, 0.35, 0.35))
    return _apply_colormap(spd, cmap, norm, mask=fields.mask)


def colorize_vorticity(
    fields: PhysicsFields,
    vorticity_range: Optional[float] = None,
) -> NDArray[np.float32]:
    """
    渦度場（正渦=逆時針=青綠，負渦=順時針=黃橘，零=黑）。
    vorticity_range: 對稱截斷值（None → 自動 98th 百分位）。
    """
    w = fields.vorticity
    if vorticity_range is None:
        fluid_w = w[fields.mask < 0.5]
        vorticity_range = float(np.percentile(np.abs(fluid_w), 98)) + 1e-6
    norm = Normalize(vmin=-vorticity_range, vmax=vorticity_range)
    cmap = _build_vorticity_cmap()
    return _apply_colormap(w, cmap, norm, mask=fields.mask)


def colorize_pressure(
    fields: PhysicsFields,
    p_range: Optional[float] = None,
) -> NDArray[np.float32]:
    """
    壓力偏差場（紅=高壓，藍=低壓，白=平均）。
    """
    p = fields.pressure
    if p_range is None:
        fluid_p = p[fields.mask < 0.5]
        p_range = float(np.percentile(np.abs(fluid_p), 98)) + 1e-6
    norm = Normalize(vmin=-p_range, vmax=p_range)
    cmap = cm.get_cmap("RdBu_r")
    cmap.set_bad(color=(0.35, 0.35, 0.35))
    return _apply_colormap(p, cmap, norm, mask=fields.mask)


def colorize_stress(
    fields: PhysicsFields,
    stress_range: Optional[float] = None,
) -> NDArray[np.float32]:
    """
    剪切應力 pxy（coolwarm，建築結構風壓參考用）。
    """
    pxy = fields.pxy.copy()
    pxy[fields.mask > 0.5] = 0.0
    if stress_range is None:
        fluid_pxy = pxy[fields.mask < 0.5]
        stress_range = float(np.percentile(np.abs(fluid_pxy), 98)) + 1e-6
    norm = Normalize(vmin=-stress_range, vmax=stress_range)
    cmap = cm.get_cmap("coolwarm")
    cmap.set_bad(color=(0.35, 0.35, 0.35))
    return _apply_colormap(pxy, cmap, norm, mask=fields.mask)


def colorize_beaufort(
    fields: PhysicsFields,
) -> NDArray[np.float32]:
    """
    蒲氏風力級數離散上色。
    固體像素 → 深灰；0-12 級 → 各自代表色。
    """
    spd_ms = fields.speed_ms
    cmap, norm = _build_beaufort_cmap()
    cmap.set_bad(color=(0.35, 0.35, 0.35))

    # 固體設 NaN
    plot_spd = spd_ms.copy()
    plot_spd[fields.mask > 0.5] = np.nan

    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba = mapper.to_rgba(plot_spd)
    rgb  = rgba[:, :, :3].astype(np.float32)
    rgb[fields.mask > 0.5] = 0.35       # 固體深灰

    return rgb


# ══════════════════════════════════════════════════════════════════════════════
# 統一分派器
# ══════════════════════════════════════════════════════════════════════════════

VIZ_MODES = ["velocity", "vorticity", "pressure", "beaufort", "stress"]

def colorize(
    state_np: NDArray[np.float32],
    mode: str = "velocity",
    lbm_to_ms: float = 10.0,
    **kwargs,
) -> NDArray[np.float32]:
    """
    統一入口：從 NCA state [11, H, W] 或 [1, 11, H, W] 產生 [H, W, 3] RGB。

    Args:
        state_np  : NCA 輸出的前 11 通道（含 mask/SDF + 9 moments）
        mode      : "velocity" | "vorticity" | "pressure" | "beaufort" | "stress"
        lbm_to_ms : LBM 單位 → m/s 換算係數（Beaufort 需要 m/s）
        **kwargs  : 傳遞給各個 colorize_* 函數的選用參數

    Returns:
        [H, W, 3] float32 RGB，值域 [0, 1]
    """
    fields = PhysicsFields(state_np, lbm_to_ms=lbm_to_ms)

    if mode == "velocity":
        return colorize_velocity(fields, **kwargs)
    elif mode == "vorticity":
        return colorize_vorticity(fields, **kwargs)
    elif mode == "pressure":
        return colorize_pressure(fields, **kwargs)
    elif mode == "beaufort":
        return colorize_beaufort(fields)
    elif mode == "stress":
        return colorize_stress(fields, **kwargs)
    else:
        raise ValueError(f"未知的視覺化模式: '{mode}'，可用: {VIZ_MODES}")


def get_beaufort_legend() -> list[dict]:
    """
    回傳蒲氏級數圖例列表，供前端/GH 顯示色卡說明。

    Returns:
        [{"level": 0, "label": "0 靜風", "color_rgb": [0.90, 0.97, 1.00], "max_ms": 0.3}, ...]
    """
    legend = []
    for i, (label, color, upper) in enumerate(
        zip(BEAUFORT_LABELS, BEAUFORT_COLORS, BEAUFORT_LIMITS_MS)
    ):
        legend.append({
            "level":    i,
            "label":    label,
            "color_rgb": list(color),
            "color_hex": "#{:02X}{:02X}{:02X}".format(
                int(color[0]*255), int(color[1]*255), int(color[2]*255)
            ),
            "max_ms":   upper if upper < 999 else None,
        })
    return legend
