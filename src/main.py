# 匯入必要套件
import taichi as ti
import asyncio
import websockets
import json
import logging
import numpy as np
import yaml

# --- 1. 讀取設定檔 ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- 2. 初始化 Taichi ---
ti.init(arch=ti.gpu)

# --- 3. 匯入自定義引擎 ---
from engine import ComputeEngine
from engine_sdf import SDFEngine

# --- 4. 設定日誌 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("RhinoBridge")

# --- 5. 初始化計算引擎 ---
engine_config = {
    "data_res": config["data_res"],
    "gui_res": config["gui_res"],
    "max_edges": config["max_edges"],
}
engine_mask = ComputeEngine(**engine_config)
engine_sdf = SDFEngine(**engine_config)

# 建立統一的 GUI 視窗
gui = ti.GUI("Rhino Taichi Bridge", res=config["gui_res"])

# --- 6. 全域狀態 ---
current_mode = "sdf"  # 預設模式
latest_polygons = []  # 快取最新的幾何數據


async def ws_handler(websocket):
    """處理 WebSocket 連線和訊息的非同步函式"""
    global current_mode, latest_polygons
    logger.info("客戶端已連線")

    try:
        # 持續接收客戶端訊息
        async for message in websocket:
            try:
                data = json.loads(message)

                # 標記是否需要刷新引擎
                need_update = False

                # 1. 處理幾何數據更新 (更新快取)
                if "polygons" in data:
                    latest_polygons = data["polygons"]
                    need_update = True

                # 2. 處理模式切換
                if "mode" in data:
                    new_mode = data["mode"]
                    if new_mode != current_mode:
                        logger.info(f"切換模式: {current_mode} -> {new_mode}")
                        current_mode = new_mode
                        # 模式改變時，必須強制刷新，讓新引擎計算現有幾何
                        need_update = True

                # 3. 統一執行引擎更新
                # 只要有收到多邊形 OR 切換了模式，且快取中有數據，就執行計算
                if need_update and latest_polygons:
                    if current_mode == "sdf":
                        engine_sdf.update_geometry(latest_polygons)
                    else:
                        engine_mask.update_geometry(latest_polygons, {})

                # 回傳狀態確認
                await websocket.send(json.dumps({"status": "ok", "mode": current_mode}))

            except Exception as e:
                logger.error(f"處理訊息時發生錯誤: {e}")

    except websockets.exceptions.ConnectionClosed:
        logger.info("客戶端已斷線")


async def render_loop():
    """統一的渲染迴圈，負責更新 GUI 視窗"""
    global current_mode
    while gui.running:
        # 根據模式選擇要顯示的畫布
        if current_mode == "sdf":
            gui.set_image(engine_sdf.canvas)
            gui.text(
                "Mode: SDF (Signed Distance Field)",
                pos=(0.05, 0.95),
                color=0xFFFFFF,
                font_size=20,
            )
        else:
            gui.set_image(engine_mask.canvas)
            gui.text(
                "Mode: Binary Mask (Union)",
                pos=(0.05, 0.95),
                color=0xFFFFFF,
                font_size=20,
            )

        gui.show()
        await asyncio.sleep(0.01)


async def main():
    """主函式，啟動伺服器和渲染迴圈"""
    ws_host = config["ws_host"]
    ws_port = config["ws_port"]

    server = await websockets.serve(ws_handler, ws_host, ws_port)
    logger.info(f"伺服器已啟動於 ws://{ws_host}:{ws_port}")

    await render_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
