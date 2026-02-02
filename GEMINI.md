### 專案概觀

本專案是一個 WebSocket 伺服器，作為客戶端應用程式（很可能是 Rhino）和基於 Taichi 的計算後端之間的橋樑。它從客戶端接收多邊形數據，使用二元遮罩或符號距離場（SDF）算法在 GPU 上進行處理，並即時視覺化結果。

**關鍵技術：**

- **Python：** 核心後端邏輯是使用 Python 編寫的。
- **Taichi：** 一個基於 Python 的平行計算框架，用於在 GPU 上進行高效能數值運算。
- **WebSockets：** 用於客戶端和伺服器之間的即時通訊。在 Python 中使用 `websockets` 函式庫，在 Node.js 中使用 `ws` 函式庫。
- **Node.js：** 使用 Node.js 腳本來運行 Python 腳本。
- **Numpy：** 用於數值運算。

**架構：**

該應用程式由三個主要部分組成：

1.  **WebSocket 伺服器（`src/main.py`）：** 這是應用程式的主要進入點。它處理 WebSocket 連線，從客戶端接收數據，並根據當前模式將其分派到適當的引擎。
2.  **計算引擎（`src/engine.py`、`src/engine_sdf.py`）：** 這些模組包含處理多邊形數據的核心邏輯。
    - `ComputeEngine`（`engine.py`）：使用環繞數算法計算輸入多邊形的二元遮罩。
    - `SDFEngine`（`engine_sdf.py`）：計算輸入多邊形的符號距離場（SDF）。
3.  **客戶端（`src/client.py`）：** 一個用於測試伺服器的簡單 Python WebSocket 客戶端。主要客戶端可能是一個 Rhino 插件。

### 建置與執行

要運行此專案，您需要安裝 Python、Node.js 和所需的相依套件。

**安裝：**

1.  **Python 相依套件：**
    ```bash
    pip install taichi websockets numpy
    ```
2.  **Node.js 相依套件：**
    ```bash
    npm install
    ```

**執行伺服器：**

```bash
npm test
```

此命令將在 `ws://localhost:8765` 上啟動 WebSocket 伺服器。

### 開發慣例

- 程式碼是使用 Python 編寫的，並遵循 PEP 8 樣式指南。
- 使用 `taichi` 函式庫進行 GPU 加速計算。
- 使用 `websockets` 函式庫進行 WebSocket 通訊。
- 專案採用模組化架構，核心邏輯分散在不同的模組中。
- `main.py` 腳本使用 `asyncio` 來並行執行 WebSocket 伺服器和渲染循環。
