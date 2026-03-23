"""
nca_loader.py
=============
跨 repo 載入 02-nca-cfd 訓練好的 .pth，回傳可直接推論的 model。

支援：
  - VAENCA         (新) : checkpoint["config"] 含 "VAENCA" 區段
  - LBMInspiredNCA (舊) : checkpoint["config"] 含 "Model"  區段（向下相容）

統一入口 → load_model(checkpoint_path)
回傳 ModelInfo TypedDict，供 main.py 與 NCAEngine 使用。
"""

import os
import sys
import logging
from typing import TypedDict

import torch
import torch.nn as nn

logger = logging.getLogger("RhinoBridge")


# ─────────────────────────────────────────────────────────────────────────────
# 回傳格式定義
# ─────────────────────────────────────────────────────────────────────────────

class ModelInfo(TypedDict):
    model:           nn.Module   # eval mode，已搬到正確 device
    model_type:      str         # "vaenca" | "lbm_inspired"
    total_channels:  int         # NCA 狀態總通道數 (VAENCA=32, LBM 依 config)
    static_channels: int         # 靜態通道 (通常=2)
    phy_channels:    int         # 物理矩通道 (通常=9)


# ─────────────────────────────────────────────────────────────────────────────
# 路徑工具
# ─────────────────────────────────────────────────────────────────────────────

def _find_nca_cfd_src(checkpoint_path: str, override: str | None = None) -> str:
    """
    從 checkpoint 路徑向上搜尋 02-nca-cfd/src。
    預期 monorepo 結構：
        NCA_workspace/
            02-nca-cfd/
                src/
                train_log/.../vaenca_model_final_*.pth
            03-gh-frontend/
                src/
    """
    if override:
        return override

    ckpt_abs  = os.path.abspath(checkpoint_path)
    candidate = os.path.dirname(ckpt_abs)
    for _ in range(10):
        candidate = os.path.dirname(candidate)
        maybe = os.path.join(candidate, "02-nca-cfd", "src")
        if os.path.isdir(maybe):
            return maybe

    raise FileNotFoundError(
        f"找不到 02-nca-cfd/src。請在 config.yaml 的 nca_cfd_src 手動指定路徑。\n"
        f"搜尋起點：{ckpt_abs}"
    )


def _inject_src(src_path: str) -> None:
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        logger.info(f"[Loader] sys.path += {src_path}")


# ─────────────────────────────────────────────────────────────────────────────
# VAENCA 載入
# ─────────────────────────────────────────────────────────────────────────────

def _load_vaenca(
    checkpoint: dict,
    device:     torch.device,
    src_path:   str,
) -> ModelInfo:
    """從已解析的 checkpoint dict 建立 VAENCA 並載入權重。"""
    _inject_src(src_path)
    from models.VAENCA import VAENCA  # noqa: E402  (動態注入後才可 import)

    vaenca_cfg = checkpoint["config"]["VAENCA"]

    # 補齊舊版 checkpoint 可能缺少的欄位
    vaenca_cfg.setdefault("static_channels",   2)
    vaenca_cfg.setdefault("phy_channels",      9)
    vaenca_cfg.setdefault("hidden_channels",  21)
    vaenca_cfg.setdefault("hidden_dim",       32)
    vaenca_cfg.setdefault("nca_hidden",      128)
    vaenca_cfg.setdefault("learnable_filters", 16)
    vaenca_cfg.setdefault("hormone_dim",        4)
    vaenca_cfg.setdefault("fire_rate",        0.5)

    model = VAENCA(vaenca_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    static_ch = int(vaenca_cfg["static_channels"])
    phy_ch    = int(vaenca_cfg["phy_channels"])
    total_ch  = model.total_channels

    logger.info(
        f"[Loader] ✅ VAENCA | step={checkpoint.get('step','?')} | "
        f"total_ch={total_ch} | static={static_ch} | phy={phy_ch} | device={device}"
    )
    return ModelInfo(
        model=model,
        model_type="vaenca",
        total_channels=total_ch,
        static_channels=static_ch,
        phy_channels=phy_ch,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LBMInspiredNCA 載入（向下相容）
# ─────────────────────────────────────────────────────────────────────────────

def _load_lbm_inspired(
    checkpoint: dict,
    device:     torch.device,
    src_path:   str,
) -> ModelInfo:
    """從已解析的 checkpoint dict 建立 LBMInspiredNCA 並載入權重。"""
    _inject_src(src_path)
    from models.LBMInspiredNCA import LBMInspiredNCA  # noqa: E402

    cfg       = checkpoint["config"]
    model_cfg = cfg.get("Model", {})
    pool_cfg  = cfg.get("RollingPool", {})

    model_cfg.setdefault("nca_channels",      32)
    model_cfg.setdefault("hidden_n",          96)
    model_cfg.setdefault("learnable_filters", 16)
    model_cfg.setdefault("num_groups",         8)
    static_ch = int(pool_cfg.get("static_channels", 2))

    model = LBMInspiredNCA(model_cfg, static_channels=static_ch)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    total_ch = int(model_cfg["nca_channels"])
    logger.info(
        f"[Loader] ✅ LBMInspiredNCA | step={checkpoint.get('step','?')} | "
        f"total_ch={total_ch} | static={static_ch} | device={device}"
    )
    return ModelInfo(
        model=model,
        model_type="lbm_inspired",
        total_channels=total_ch,
        static_channels=static_ch,
        phy_channels=9,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 公開統一入口
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str,
    nca_cfd_src:     str | None = None,
    device:          torch.device | None = None,
) -> ModelInfo:
    """
    自動偵測 checkpoint 類型並載入對應模型。

    Args:
        checkpoint_path : .pth 絕對路徑
        nca_cfd_src     : 02-nca-cfd/src 路徑（None → 從路徑自動推算）
        device          : None → 自動選 CUDA / CPU

    Returns:
        ModelInfo TypedDict（model, model_type, total_channels,
                             static_channels, phy_channels）
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"[Loader] 載入 checkpoint：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    src_path = _find_nca_cfd_src(checkpoint_path, nca_cfd_src)
    cfg      = checkpoint.get("config", {})

    if "VAENCA" in cfg:
        return _load_vaenca(checkpoint, device, src_path)
    elif "Model" in cfg:
        return _load_lbm_inspired(checkpoint, device, src_path)
    else:
        raise KeyError(
            "[Loader] checkpoint config 中找不到 'VAENCA' 或 'Model' 區段。\n"
            f"現有 keys：{list(cfg.keys())}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 舊版函式（向下相容，勿刪）
# ─────────────────────────────────────────────────────────────────────────────

def load_nca_model(checkpoint_path: str, nca_cfd_src: str = None):
    """舊版 API，保留向下相容。回傳 (model, fake_cfg, static_ch)。"""
    info     = load_model(checkpoint_path, nca_cfd_src)
    fake_cfg = {"nca_channels": info["total_channels"]}
    return info["model"], fake_cfg, info["static_channels"]
