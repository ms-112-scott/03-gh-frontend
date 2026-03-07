"""
nca_loader.py
跨 repo 載入 02-nca-cfd 訓練好的 .pth，回傳可直接推論的 model。

放置於：NCA_workspace/03-gh-frontend/src/nca_loader.py
"""
import sys
import os
import torch
import logging

logger = logging.getLogger("RhinoBridge")


def load_nca_model(checkpoint_path: str, nca_cfd_src: str = None):
    """
    從 02-nca-cfd 的 checkpoint 載入 LBMInspiredNCA。

    Args:
        checkpoint_path : .pth 的絕對路徑
        nca_cfd_src     : 02-nca-cfd/src 的路徑。
                          None → 自動從 monorepo 根目錄推算

    Returns:
        model           : eval() 模式，已搬到正確 device
        model_cfg       : 原始 config["Model"] 字典
        static_channels : int (通常 = 2)
    """

    # ── 1. 找到並注入 02-nca-cfd/src ──────────────────────────────────────
    if nca_cfd_src is None:
        # 從 checkpoint 路徑往上找到 NCA_workspace 根目錄
        # 預期結構：NCA_workspace/02-nca-cfd/train_log/MMDD_N/model_latest.pth
        ckpt_abs = os.path.abspath(checkpoint_path)
        candidate = os.path.dirname(ckpt_abs)
        found = False
        for _ in range(8):
            candidate = os.path.dirname(candidate)
            maybe = os.path.join(candidate, "02-nca-cfd", "src")
            if os.path.isdir(maybe):
                nca_cfd_src = maybe
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"找不到 02-nca-cfd/src，請手動設定 nca_cfd_src 參數。\n"
                f"從 checkpoint 路徑向上搜尋: {ckpt_abs}"
            )

    if nca_cfd_src not in sys.path:
        sys.path.insert(0, nca_cfd_src)
        logger.info(f"[Loader] sys.path 已加入: {nca_cfd_src}")

    # ── 2. 載入 checkpoint ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    logger.info(f"[Loader] 讀取 checkpoint: {checkpoint_path}  (step={checkpoint.get('step', '?')})")

    # ── 3. 解析 config ──────────────────────────────────────────────────
    config = checkpoint.get("config", {})
    model_cfg = config.get("Model", {})
    pool_cfg  = config.get("RollingPool", {})

    # 補齊預設值（對應 config_desktop.yaml 的常見設定）
    model_cfg.setdefault("nca_channels",      32)
    model_cfg.setdefault("hidden_n",          96)
    model_cfg.setdefault("learnable_filters", 16)
    model_cfg.setdefault("num_groups",         8)
    static_channels = pool_cfg.get("static_channels", 2)

    # ── 4. 建立模型並載入權重 ───────────────────────────────────────────
    from models.LBMInspiredNCA import LBMInspiredNCA  # noqa

    model = LBMInspiredNCA(model_cfg, static_channels=static_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        f"[Loader] ✅ 模型載入成功 | "
        f"channels={model_cfg['nca_channels']} | "
        f"hidden={model_cfg['hidden_n']} | "
        f"static_ch={static_channels} | "
        f"device={device}"
    )
    return model, model_cfg, static_channels
