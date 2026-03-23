# 03-gh-frontend Repo 結構規劃

## 文件範圍

- 本文件已讀取並參考 `02-nca-cfd/docs/07_MVC_Integration/` 底下的 MVC 規劃。
- 本 repo 只處理 `03-gh-frontend` 端的職責。
- `02-nca-cfd` 相關檔案視為唯讀參考與依賴來源，不在本 repo 內修改。

## 與 MVC 規劃的對齊整理

### 對齊 `01_Model_ML_Backend.md`

- `03-gh-frontend` 需要把幾何資料與 7 維條件參數整理成後端可接受的輸入格式。
- 前端只負責載入與驅動模型，不負責模型訓練、權重管理與後端正規化資產本身。
- repo 結構上應把模型載入、執行控制、視覺化與 GH 傳輸層分開。

### 對齊 `02_View_Taichi_Middleware.md`

- Taichi middleware 應負責幾何 rasterize、mask / SDF 生成與物理場視覺化前處理。
- `viz_mode`、Beaufort 圖例與顏色映射屬於 frontend 的責任。
- 流程應維持清楚可讀：
  幾何輸入 -> Taichi 前處理 -> NCA 輸出 -> 色彩映射 -> GUI / WebSocket 回傳

### 對齊 `03_Controller_GH_Client.md`

- Grasshopper client 應視為 controller / transport 層，不應混入 Taichi 或模型核心邏輯。
- WebSocket payload、重啟控制、viz 切換與狀態文字應與運算模組分離。
- repo 結構應反映 GH Controller、Taichi Middleware、NCA Integration 三層邊界。

## 目前已重整後的結構

```text
03-gh-frontend/
├─ doc/
│  ├─ repo-structure-plan.md
│  └─ 使用說明.md
├─ scripts/
│  └─ export_code_base.py
├─ src/
│  ├─ app/
│  │  ├─ __init__.py
│  │  └─ main.py
│  ├─ gh/
│  │  ├─ __init__.py
│  │  └─ grasshopper_client.py
│  ├─ integration/
│  │  ├─ __init__.py
│  │  ├─ nca_engine.py
│  │  └─ nca_loader.py
│  ├─ middleware/
│  │  ├─ __init__.py
│  │  ├─ mask_engine.py
│  │  └─ sdf_engine.py
│  ├─ tools/
│  │  ├─ __init__.py
│  │  ├─ cv2_test.py
│  │  └─ ws_client_demo.py
│  ├─ visualization/
│  │  ├─ __init__.py
│  │  └─ physics_colorizer.py
│  └─ __init__.py
├─ .gitignore
├─ config.yaml
├─ requirements.txt
├─ package.json
└─ package-lock.json
```

## 各資料夾職責

### `src/app`

- 專案執行入口
- WebSocket server 啟動
- GUI render loop 協調
- 各模式切換與主流程控制

### `src/gh`

- Grasshopper GHPython 腳本
- 幾何、bounds、mode、viz_mode 的 payload 組裝
- WebSocket 回應解析
- 狀態文字與圖例整理
- 選用的 JPEG 輸出處理

### `src/middleware`

- 幾何資料轉換為 mask / SDF
- Taichi 前處理與場資料準備
- 將 frontend 的幾何輸入整理為可供 NCA 使用的中介資料

### `src/integration`

- 從唯讀的 `02-nca-cfd` 載入 checkpoint
- 把模型包裝為 frontend 可驅動的執行引擎
- 將執行結果轉成統計資料、圖像與圖例

### `src/visualization`

- 物理場解碼與顏色映射
- Beaufort 分級與圖例生成
- `viz_mode` 相關視覺化規則

### `src/tools`

- 本地測試與工具腳本
- 環境檢查
- 非正式 demo 或除錯用途程式

### `scripts`

- 專案維護用腳本
- 匯出 codebase、資料盤點或自動化輔助工具

## 建議中的目標 repo 結構

```text
03-gh-frontend/
├─ doc/
│  ├─ repo-structure-plan.md
│  ├─ 使用說明.md
│  ├─ architecture-overview.md
│  ├─ websocket-contract.md
│  └─ setup-and-run.md
├─ scripts/
│  └─ export_code_base.py
├─ src/
│  ├─ app/
│  │  ├─ bootstrap.py
│  │  ├─ config.py
│  │  ├─ main.py
│  │  └─ runtime.py
│  ├─ gh/
│  │  ├─ client.py
│  │  ├─ payloads.py
│  │  └─ response_formatters.py
│  ├─ integration/
│  │  ├─ nca_engine.py
│  │  ├─ nca_loader.py
│  │  └─ contracts.py
│  ├─ middleware/
│  │  ├─ mask_engine.py
│  │  ├─ sdf_engine.py
│  │  └─ geometry_types.py
│  ├─ visualization/
│  │  ├─ physics_colorizer.py
│  │  ├─ legends.py
│  │  └─ viz_modes.py
│  ├─ tools/
│  │  ├─ cv2_test.py
│  │  └─ ws_client_demo.py
│  └─ tests/
│     ├─ test_payloads.py
│     ├─ test_viz_modes.py
│     └─ test_geometry_preprocess.py
├─ assets/
│  ├─ images/
│  └─ models/
├─ .gitignore
├─ config.yaml
├─ requirements.txt
├─ package.json
└─ README.md
```

## 後續建議整理步驟

- 將 `main.py` 中的設定讀取拆到 `src/app/config.py`
- 將 WebSocket payload 處理與 render loop 再進一步拆開
- 新增 payload schema、viz mode、geometry normalization 的測試
- 補齊 controller contract、執行流程與架構說明文件
