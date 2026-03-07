# main.py  (視覺化升級版)
# 在原有 mask / sdf / nca 模式基礎上，新增 viz_mode 切換
#
# config.yaml 新增欄位：
#   nca_checkpoint: "../../02-nca-cfd/train_log/MMDD_N/model_latest.pth"
#   nca_steps_per_frame: 4
#   lbm_to_ms: 10.0          # LBM u=1.0 對應幾 m/s（影響 Beaufort 換算）

import taichi as ti
import asyncio
import websockets
import json
import logging
import numpy as np
import yaml

# ── 1. 讀取設定檔 ─────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ── 2. 統一初始化 Taichi ─────────────────────────────────────────────────
ti.init(arch=ti.gpu)

# ── 3. 匯入引擎 ──────────────────────────────────────────────────────────
from engine import ComputeEngine
from engine_sdf import SDFEngine
from engine_nca import NCAEngine          # engine_nca.py (視覺化升級版)
from nca_loader import load_nca_model

# ── 4. 設定日誌 ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("RhinoBridge")

# ── 5. 幾何引擎 ──────────────────────────────────────────────────────────
engine_config = {
    "data_res":  config["data_res"],
    "gui_res":   config["gui_res"],
    "max_edges": config["max_edges"],
}
engine_mask = ComputeEngine(**engine_config)
engine_sdf  = SDFEngine(**engine_config)

# ── 6. NCA 引擎 ──────────────────────────────────────────────────────────
engine_nca: NCAEngine | None = None

if ckpt := config.get("nca_checkpoint"):
    try:
        model, model_cfg, static_ch = load_nca_model(ckpt)
        engine_nca = NCAEngine(
            model=model,
            nca_channels=model_cfg["nca_channels"],
            static_channels=static_ch,
            data_res=config["data_res"],
            gui_res=config["gui_res"],
            lbm_to_ms=config.get("lbm_to_ms", 10.0),
            viz_mode=config.get("default_viz_mode", "velocity"),
        )
        logger.info("[Main] ✅ NCA 引擎初始化成功")
    except Exception as e:
        logger.error(f"[Main] ❌ NCA 引擎初始化失敗: {e}")
else:
    logger.warning("[Main] 未設定 nca_checkpoint，NCA 模式不可用。")

# ── 7. GUI ───────────────────────────────────────────────────────────────
gui = ti.GUI("Rhino NCA Bridge", res=config["gui_res"])

# ── 8. 全域狀態 ──────────────────────────────────────────────────────────
current_mode        = "sdf"     # "sdf" | "mask" | "nca"
latest_polygons     = []
nca_steps_per_frame = config.get("nca_steps_per_frame", 4)

# viz_mode 標籤（給 GUI 顯示）
VIZ_MODE_LABELS = {
    "velocity":  "速度量值",
    "vorticity": "渦度場",
    "pressure":  "壓力場",
    "beaufort":  "蒲氏風力",
    "stress":    "剪切應力",
}


# ────────────────────────────────────────────────────────────────────────
# WebSocket 訊息處理
# ────────────────────────────────────────────────────────────────────────
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

                # ── B. 主模式切換（sdf / mask / nca）────────────────────
                if "mode" in data:
                    new_mode = data["mode"]
                    if new_mode == "nca" and engine_nca is None:
                        await websocket.send(json.dumps({
                            "status": "error",
                            "msg": "NCA engine not available (missing checkpoint)",
                            "mode": current_mode,
                        }))
                        continue
                    if new_mode != current_mode:
                        logger.info(f"主模式: {current_mode} → {new_mode}")
                        current_mode = new_mode
                        need_geo_update = True  # 切換模式時強制以現有幾何刷新

                # ── C. 視覺化模式切換（viz_mode，只在 NCA 模式有效）─────
                if "viz_mode" in data and engine_nca is not None:
                    try:
                        engine_nca.set_viz_mode(data["viz_mode"])
                    except ValueError as e:
                        logger.warning(f"[WS] 無效 viz_mode: {e}")

                # ── D. 執行幾何計算 ──────────────────────────────────────
                if need_geo_update and latest_polygons:
                    if current_mode == "sdf":
                        engine_sdf.update_geometry(latest_polygons)

                    elif current_mode == "mask":
                        engine_mask.update_geometry(latest_polygons, {})

                    elif current_mode == "nca" and engine_nca is not None:
                        # 同時更新兩個幾何引擎，取出 numpy 陣列後重置 NCA seed
                        engine_mask.update_geometry(latest_polygons, {})
                        engine_sdf.update_geometry(latest_polygons)
                        mask_np = engine_mask.mask_field.to_numpy()
                        sdf_np  = engine_sdf.sdf_field.to_numpy()
                        engine_nca.reset_state(mask_np, sdf_np)

                # ── E. 回傳確認 ──────────────────────────────────────────
                response = {"status": "ok", "mode": current_mode}

                if current_mode == "nca" and engine_nca is not None and engine_nca._ready:
                    stats = engine_nca.get_velocity_stats()
                    response["stats"]    = stats
                    response["viz_mode"] = engine_nca.viz_mode

                    # 每次幾何更新時附帶圖例（方便 GH 首次顯示色卡）
                    if need_geo_update:
                        response["beaufort_legend"] = engine_nca.beaufort_legend()

                await websocket.send(json.dumps(response))

            except Exception as e:
                logger.error(f"處理訊息時發生錯誤: {e}", exc_info=True)

    except websockets.exceptions.ConnectionClosed:
        logger.info("客戶端已斷線")


# ────────────────────────────────────────────────────────────────────────
# 渲染迴圈
# ────────────────────────────────────────────────────────────────────────
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

                # 標題顯示目前的視覺化模式
                viz_label = VIZ_MODE_LABELS.get(engine_nca.viz_mode, engine_nca.viz_mode)
                stats = engine_nca.get_velocity_stats()
                bft   = stats.get("beaufort_max", 0)
                spd   = stats.get("speed_max_ms", 0.0)
                gui.text(
                    f"NCA | {viz_label} | {spd:.1f} m/s | Bft {bft}",
                    pos=(0.02, 0.97), color=0xFFFF00, font_size=18,
                )
            else:
                gui.set_image(engine_sdf.canvas)
                gui.text("NCA: 等待幾何輸入...", pos=(0.02, 0.97),
                         color=0xFF8800, font_size=18)

        gui.show()
        await asyncio.sleep(0.01)


# ────────────────────────────────────────────────────────────────────────
# 主函式
# ────────────────────────────────────────────────────────────────────────
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
