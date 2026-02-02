# 匯入必要的函式庫
import taichi as ti  # 用於 GPU 計算和 GUI
import asyncio  # 用於非同步 I/O，處理並行的網路和渲染
import websockets  # 用於建立 WebSocket 伺服器
import json  # 用於解析客戶端傳來的 JSON 訊息
import logging  # 用於記錄伺服器狀態和錯誤
import time  # 時間相關功能 (此處未使用)
import numpy as np  # 用於數值計算 (主要用於和 Taichi 交換數據)

# --- 1. 統一初始化 Taichi ---
# ti.init() 是 Taichi 的啟動點，必須在使用任何 Taichi 功能前呼叫
# arch=ti.gpu 指定計算後端為 GPU (例如 CUDA 或 Vulkan)，以獲得最佳效能
ti.init(arch=ti.gpu)

# 從其他檔案匯入我們自定義的計算引擎
from engine import ComputeEngine  # 匯入原始的二元遮罩 (Mask) 引擎
from engine_sdf import SDFEngine  # 匯入新的符號距離場 (SDF) 引擎

# --- 設定日誌記錄 ---
# basicConfig 用於設定日誌的基本格式
logging.basicConfig(
    level=logging.INFO,  # 設定日誌級別為 INFO，表示只記錄 INFO, WARNING, ERROR, CRITICAL 等級的訊息
    format="%(asctime)s [%(levelname)s] %(message)s"  # 設定日誌訊息的格式，包含時間、級別和訊息內容
)
# 建立一個名為 "RhinoBridge" 的日誌記錄器
logger = logging.getLogger("RhinoBridge")

# --- 初始化兩個引擎 ---
# 注意：這兩個引擎會共享同一個由 ti.init() 建立的 Taichi "Context"。
# 這表示它們可以在同一個 GPU 上運作，而不需要重新初始化。
engine_mask = ComputeEngine(data_res=1024, gui_res=512)  # 建立遮罩引擎的實例，設定計算解析度和顯示解析度
engine_sdf = SDFEngine(data_res=1024, gui_res=512)  # 建立 SDF 引擎的實例

# --- 建立一個統一的 GUI 視窗 ---
# ti.GUI() 建立一個 Taichi 的圖形使用者介面視窗
# "Rhino Taichi Bridge" 是視窗的標題
# res=512 設定視窗的解析度為 512x512 像素
gui = ti.GUI("Rhino Taichi Bridge", res=512)

# --- 全域狀態變數 ---
# 用於在不同的非同步函式之間共享狀態
current_mode = "sdf"  # 預設模式為 'sdf'。可以是 'mask' 或 'sdf'
latest_polygons = []  # 用於儲存從客戶端收到的最新多邊形幾何數據


# --- WebSocket 訊息處理函式 ---
# 當有客戶端連線時，這個非同步函式會被呼叫
async def ws_handler(websocket):
    # 使用 global 關鍵字來修改全域變數
    global current_mode, latest_polygons
    logger.info("客戶端已連線")

    try:
        # 使用 async for 迴圈來持續等待並接收來自客戶端的訊息
        async for message in websocket:
            try:
                # 將收到的 JSON 字串訊息解析成 Python 的字典
                data = json.loads(message)

                # 1. 檢查訊息中是否包含 'mode' 鍵，用於切換計算模式
                if "mode" in data:
                    new_mode = data["mode"]  # 取得新的模式字串
                    if new_mode != current_mode:  # 如果新模式和當前模式不同
                        logger.info(f"切換模式: {current_mode} -> {new_mode}")
                        current_mode = new_mode  # 更新全域的模式狀態

                # 2. 檢查訊息中是否包含 'polygons' 鍵，用於更新幾何數據
                if "polygons" in data:
                    polygons = data["polygons"]  # 取得多邊形列表
                    latest_polygons = polygons  # 更新全域的幾何數據快取

                    # 3. 根據當前的模式，將幾何數據分派給對應的引擎進行計算
                    if current_mode == "sdf":
                        engine_sdf.update_geometry(polygons)  # 呼叫 SDF 引擎的更新函式
                    else:
                        engine_mask.update_geometry(polygons, {})  # 呼叫遮罩引擎的更新函式

                # 向客戶端回傳一個確認訊息，告知目前狀態和模式
                await websocket.send(json.dumps({"status": "ok", "mode": current_mode}))

            except Exception as e:
                # 如果在處理單一訊息時發生錯誤，記錄錯誤但保持連線
                logger.error(f"處理訊息時發生錯誤: {e}")

    except websockets.exceptions.ConnectionClosed:
        # 如果客戶端斷開連線，記錄訊息
        logger.info("客戶端已斷線")


# --- 圖形渲染迴圈 ---
async def render_loop():
    """統一的渲染循環，負責將計算結果畫在 GUI 視窗上"""
    global current_mode
    # 當 GUI 視窗沒有被關閉時，持續執行
    while gui.running:
        # 根據當前的模式決定要渲染哪個引擎的畫布
        if current_mode == "sdf":
            # 從 SDF 引擎取得渲染好的畫布 (一個 Taichi field)
            gui.set_image(engine_sdf.canvas)
            # 在 GUI 視窗的左上角顯示目前的模式文字
            gui.text(
                "模式: SDF (符號距離場)",
                pos=(0.05, 0.95),  # 位置 (x, y)，範圍是 0.0 到 1.0
                color=0xFFFFFF,  # 顏色 (白色)
                font_size=20,  # 字體大小
            )
        else:
            # 從遮罩引擎取得渲染好的畫布
            gui.set_image(engine_mask.canvas)
            # 在 GUI 視窗上顯示模式文字
            gui.text(
                "模式: 二元遮罩 (聯集)",
                pos=(0.05, 0.95),
                color=0xFFFFFF,
                font_size=20,
            )

        # 將畫布內容顯示在視窗上
        gui.show()
        # 等待一小段時間 (0.01秒)，釋放 CPU 資源，並讓其他非同步任務有機會執行
        await asyncio.sleep(0.01)


# --- 主函式 ---
async def main():
    # 建立並啟動 WebSocket 伺服器
    # websockets.serve() 會回傳一個伺服器物件
    # ws_handler 是處理每個連線的函式
    # "localhost" 和 8765 是伺服器的位址和埠號
    server = await websockets.serve(ws_handler, "localhost", 8765)
    logger.info("伺服器已啟動於 ws://localhost:8765")

    # 並行執行渲染迴圈
    # 這裡 `await` 會等待 render_loop 結束，也就是 GUI 視窗被關閉
    await render_loop()


# --- 程式進入點 ---
# 這是 Python 腳本的標準啟動方式
if __name__ == "__main__":
    try:
        # asyncio.run() 是啟動非同步程式的進入點
        # 它會執行傳入的 main() 函式，並管理整個事件迴圈
        asyncio.run(main())
    except KeyboardInterrupt:
        # 如果使用者按下 Ctrl+C，程式會在這裡捕捉到 KeyboardInterrupt 異常
        # pass 表示什麼都不做，讓程式乾淨地退出
        pass
