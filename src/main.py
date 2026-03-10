# main.py  (VAENCA 整合版)
#
# 改動：
#   - 使用 load_model() 自動偵測 VAENCA / LBMInspiredNCA
#   - NCAEngine 改由 ModelInfo TypedDict 初始化，不再硬寫通道數
#   - GUI 標題額外顯示 model_type 與 VAENCA receptor_weights
#
# config.yaml 對應欄位：
#   nca_checkpoint      : "../../02-nca-cfd/train_log/.../vaenca_model_final_*.pth"
#   nca_cfd_src         : null          # null → 自動推算；或手動填絕對路徑
#   nca_steps_per_frame : 4
#   lbm_to_ms           : 10.0
#   default_viz_mode    : "velocity"

import taichi as ti
import asyncio
import websockets
import json
import logging
import numpy as np
import yaml

# ── 1. 設定檔 ─────────────────────────────────────────────────────────────────
CONFIG_PATH = "C:/Users/GAI/Desktop/NCA_workspace/03-gh-frontend/config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# ── 2. 統一初始化 Taichi ──────────────────────────────────────────────────────
ti.init(arch=ti.gpu)

# ── 3. 匯入引擎 ───────────────────────────────────────────────────────────────
from engine     import ComputeEngine
from engine_sdf import SDFEngine
from engine_nca import NCAEngine
from nca_loader import load_model          # ← 統一入口（自動偵測 VAENCA / LBM）

# ── 4. 日誌 ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("RhinoBridge")

# ── 5. 幾何引擎 ───────────────────────────────────────────────────────────────
engine_config = {
    "data_res":  config["data_res"],
    "gui_res":   config["gui_res"],
    "max_edges": config["max_edges"],
}
engine_mask = ComputeEngine(**engine_config)
engine_sdf  = SDFEngine(**engine_config)

# ── 6. NCA 引擎（自動偵測 VAENCA / LBMInspiredNCA）──────────────────────────
engine_nca: NCAEngine | None = None

_ckpt = config.get("nca_checkpoint")
if _ckpt:
    try:
        info = load_model(
            checkpoint_path = _ckpt,
            nca_cfd_src     = config.get("nca_cfd_src"),   # None → 自動推算
        )
        engine_nca = NCAEngine(
            model           = info["model"],
            model_type      = info["model_type"],
            total_channels  = info["total_channels"],
            static_channels = info["static_channels"],
            phy_channels    = info["phy_channels"],
            data_res        = config["data_res"],
            gui_res         = config["gui_res"],
            lbm_to_ms       = config.get("lbm_to_ms", 10.0),
            viz_mode        = config.get("default_viz_mode", "velocity"),
        )
        logger.info(
            f"[Main] ✅ NCAEngine 初始化成功 | "
            f"type={info['model_type']} | total_ch={info['total_channels']}"
        )
    except Exception as e:
        logger.error(f"[Main] ❌ NCA 引擎初始化失敗: {e}", exc_info=True)
else:
    logger.warning("[Main] config 未設定 nca_checkpoint，NCA 模式不可用。")

# ── 7. GUI ────────────────────────────────────────────────────────────────────
gui = ti.GUI("Rhino NCA Bridge", res=config["gui_res"])

# ── 8. 全域狀態 ───────────────────────────────────────────────────────────────
current_mode         = "sdf"   # "sdf" | "mask" | "nca"
latest_polygons: list = []
nca_steps_per_frame  = config.get("nca_steps_per_frame", 4)

VIZ_MODE_LABELS = {
    "velocity":  "速度量值",
    "vorticity": "渦度場",
    "pressure":  "壓力場",
    "beaufort":  "蒲氏風力",
    "stress":    "剪切應力",
}


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket 訊息處理
# ─────────────────────────────────────────────────────────────────────────────

async def ws_handler(websocket):
    global current_mode, latest_polygons
    logger.info("客戶端已連線")

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                need_geo_update = False

                # ── A. 幾何更新 ──────────────────────────────────────────
                if "polygons" in data:
                    latest_polygons = data["polygons"]
                    need_geo_update = True

                # ── B. 主模式切換 ────────────────────────────────────────
                if "mode" in data:
                    new_mode = data["mode"]
                    if new_mode == "nca" and engine_nca is None:
                        await websocket.send(json.dumps({
                            "status": "error",
                            "msg":    "NCA engine not available (missing checkpoint)",
                            "mode":   current_mode,
                        }))
                        continue
                    if new_mode != current_mode:
                        logger.info(f"主模式: {current_mode} → {new_mode}")
                        current_mode    = new_mode
                        need_geo_update = True

                # ── C. 視覺化模式切換 ────────────────────────────────────
                if "viz_mode" in data and engine_nca is not None:
                    try:
                        engine_nca.set_viz_mode(data["viz_mode"])
                    except ValueError as e:
                        logger.warning(f"[WS] 無效 viz_mode: {e}")

                # ── D. 幾何計算 ──────────────────────────────────────────
                if need_geo_update and latest_polygons:
                    if current_mode == "sdf":
                        engine_sdf.update_geometry(latest_polygons)

                    elif current_mode == "mask":
                        engine_mask.update_geometry(latest_polygons, {})

                    elif current_mode == "nca" and engine_nca is not None:
                        engine_mask.update_geometry(latest_polygons, {})
                        engine_sdf.update_geometry(latest_polygons)
                        mask_np = engine_mask.mask_field.to_numpy()
                        sdf_np  = engine_sdf.sdf_field.to_numpy()
                        engine_nca.reset_state(mask_np, sdf_np)

                # ── E. 回傳 ──────────────────────────────────────────────
                response = {"status": "ok", "mode": current_mode}

                if current_mode == "nca" and engine_nca is not None and engine_nca._ready:
                    stats = engine_nca.get_velocity_stats()
                    response["stats"]    = stats
                    response["viz_mode"] = engine_nca.viz_mode

                    if need_geo_update:
                        response["beaufort_legend"] = engine_nca.beaufort_legend()

                await websocket.send(json.dumps(response))

            except Exception as e:
                logger.error(f"處理訊息時發生錯誤: {e}", exc_info=True)

    except websockets.exceptions.ConnectionClosed:
        logger.info("客戶端已斷線")


# ─────────────────────────────────────────────────────────────────────────────
# 渲染迴圈
# ─────────────────────────────────────────────────────────────────────────────

async def render_loop():
    while gui.running:

        if current_mode == "sdf":
            gui.set_image(engine_sdf.canvas)
            gui.text("Mode: SDF", pos=(0.02, 0.97), color=0xFFFFFF, font_size=18)

        elif current_mode == "mask":
            gui.set_image(engine_mask.canvas)
            gui.text("Mode: Mask", pos=(0.02, 0.97), color=0xFFFFFF, font_size=18)

        elif current_mode == "nca":
            if engine_nca is not None and engine_nca._ready:
                engine_nca.step(nca_steps_per_frame)
                gui.set_image(engine_nca.canvas)

                viz_label = VIZ_MODE_LABELS.get(engine_nca.viz_mode, engine_nca.viz_mode)
                stats     = engine_nca.get_velocity_stats()
                bft       = stats.get("beaufort_max", 0)
                spd       = stats.get("speed_max_ms", 0.0)
                mtype     = stats.get("model_type", "")

                # 第一行：模型類型 + 物理量
                gui.text(
                    f"{mtype.upper()} | {viz_label} | {spd:.1f} m/s | Bft {bft}",
                    pos=(0.02, 0.97),
                    color=0xFFFF00,
                    font_size=18,
                )

                # 第二行（VAENCA 專屬）：受體注意力權重
                if mtype == "vaenca" and "receptor_weights" in stats:
                    rw = stats["receptor_weights"]
                    gui.text(
                        f"A:{rw['A_topo']:.2f} "
                        f"B:{rw['B_stream']:.2f} "
                        f"C:{rw['C_mrt']:.2f} "
                        f"D:{rw['D_learned']:.2f} "
                        f"E:{rw['E_hormone']:.2f}",
                        pos=(0.02, 0.93),
                        color=0x88FFCC,
                        font_size=14,
                    )
            else:
                gui.set_image(engine_sdf.canvas)
                gui.text(
                    "NCA: 等待幾何輸入...",
                    pos=(0.02, 0.97),
                    color=0xFF8800,
                    font_size=18,
                )

        gui.show()
        await asyncio.sleep(0.01)


# ─────────────────────────────────────────────────────────────────────────────
# 主函式
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    ws_host = config["ws_host"]
    ws_port = config["ws_port"]
    server  = await websockets.serve(ws_handler, ws_host, ws_port)
    logger.info(f"伺服器已啟動於 ws://{ws_host}:{ws_port}")
    logger.info(f"可用 viz_mode: {NCAEngine.available_viz_modes()}")
    await render_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
