# 匯入必要的函式庫
import taichi as ti  # 用於 GPU 計算
import numpy as np  # 用於數值計算和數據交換
import asyncio      # 用於非同步程式設計 (在此檔案中未使用，但保留以保持一致性)

# 注意：在此檔案中不呼叫 ti.init()。
# 為了確保所有 Taichi 物件 (field, kernel) 都在同一個計算上下文中，
# `ti.init()` 應該在主程式 (main.py) 中統一呼叫一次。

# --- Taichi 資料導向類別 ---
@ti.data_oriented
class SDFEngine:
    # --- 初始化函式 ---
    def __init__(self, data_res=256, gui_res=512):
        # --- 成員變數定義 ---
        self.res = data_res          # SDF 計算網格的解析度
        self.gui_res = gui_res        # GUI 顯示的解析度

        # --- Taichi Field 定義 ---
        # 建立一個 2D 浮點數 field，用於儲存每個網格點的符號距離值 (Signed Distance Field)
        self.sdf_field = ti.field(dtype=ti.f32, shape=(self.res, self.res))
        # 建立一個 2D 向量 field，用於儲存最終渲染出的彩色圖像
        # ti.Vector.field(3, ...) 表示每個像素儲存一個 3 維向量 (R, G, B)
        self.canvas = ti.Vector.field(3, dtype=ti.f32, shape=(self.gui_res, self.gui_res))

        # --- 邊緣數據儲存 (與 ComputeEngine 相同) ---
        self.MAX_EDGES = 2000  # 最大邊緣數
        self.edges = ti.Vector.field(2, dtype=ti.f32, shape=(self.MAX_EDGES, 2)) # 儲存邊的兩個頂點
        self.num_edges = ti.field(dtype=ti.i32, shape=()) # 儲存當前有效的邊數

        # 此引擎不建立自己的 GUI，而是由主程式統一管理和渲染
        self.is_running = True # 狀態旗標

    # --- Taichi GPU 輔助函式 ---
    # @ti.func 裝飾器表示這是一個可以在 ti.kernel 中被呼叫的輔助函式
    # 它本身不會直接在 GPU 上啟動，而是會被內聯 (inline) 到呼叫它的核心函式中
    @ti.func
    def dist_point_segment(self, p, v0, v1):
        """計算點 p 到線段 v0-v1 的最短距離 (在 GPU 上執行)"""
        # 向量運算
        pa = p - v0  # 從 v0 指向 p 的向量
        ba = v1 - v0  # 從 v0 指向 v1 的向量 (線段本身)
        # 計算 p 在 ba 上的投影長度比例 h
        # h = (pa · ba) / |ba|^2
        # ti.max(0.0, ti.min(1.0, ...)) 將 h 的值限制在 [0, 1] 範圍內
        # 如果 h < 0，表示 p 的投影點在 v0 之外，離 p 最近的點是 v0
        # 如果 h > 1，表示 p 的投影點在 v1 之外，離 p 最近的點是 v1
        # 如果 0 <= h <= 1，表示投影點在線段上
        h = ti.max(0.0, ti.min(1.0, pa.dot(ba) / ba.dot(ba)))
        # 最短距離向量 = pa - ba * h
        # .norm() 計算向量的長度 (L2 範數)
        return (pa - ba * h).norm()

    # --- Taichi GPU 核心函式 ---
    @ti.kernel
    def compute_sdf_gpu(self):
        """
        在 GPU 上並行計算每個像素的符號距離場 (SDF)。
        SDF 的值代表該點到最近邊緣的距離，符號代表該點在圖形內部 (-) 或外部 (+)。
        """
        # 並行遍歷計算網格中的每一個像素 (i, j)
        for i, j in self.sdf_field:
            # 將像素索引轉換為 [0, 1] 範圍的標準化座標
            p = ti.Vector([i / self.res, j / self.res])

            min_dist = 1e9  # 初始化一個極大的距離值
            winding = 0       # 初始化環繞數

            # 遍歷所有有效的邊緣
            for k in range(self.num_edges[None]):
                v0 = self.edges[k, 0] # 邊的起點
                v1 = self.edges[k, 1] # 邊的終點

                # 1. 計算點 p 到目前這條邊的最短物理距離
                d = self.dist_point_segment(p, v0, v1)
                min_dist = ti.min(min_dist, d) # 更新全局最小距離

                # 2. 同時計算環繞數 (Winding Number) 來判斷點的內外
                if v0.y <= p.y < v1.y and p.x < (v1.x - v0.x) * (p.y - v0.y) / (v1.y - v0.y) + v0.x:
                    winding += 1 # 向上穿越
                elif v1.y <= p.y < v0.y and p.x < (v1.x - v0.x) * (p.y - v0.y) / (v1.y - v0.y) + v0.x:
                    winding -= 1 # 向下穿越

            # 根據環繞數決定距離的符號
            if winding != 0:
                # 如果環繞數不為 0，表示點在內部，距離為負
                self.sdf_field[i, j] = -min_dist
            else:
                # 如果環繞數為 0，表示點在外部，距離為正
                self.sdf_field[i, j] = min_dist

    # --- Taichi GPU 核心函式 ---
    @ti.kernel
    def render_isolines(self):
        """將計算好的 SDF 距離場渲染成漂亮的等高線視覺效果"""
        # 並行遍歷顯示畫布上的每一個像素
        for i, j in self.canvas:
            # 透過最近鄰插值從 SDF 計算網格中取樣
            sdf_val = self.sdf_field[i * self.res // self.gui_res, j * self.res // self.gui_res]

            # 初始化顏色為黑色
            color = ti.Vector([0.0, 0.0, 0.0])

            # --- 視覺化邏輯 ---
            # 使用 sin 函數根據 SDF 值產生波浪效果，形成等高線
            # 頻率 150.0 決定了線條的密度
            wave = ti.sin(sdf_val * 150.0)

            if sdf_val < 0:  # 內部
                # 基礎顏色為藍色調
                base_blue = ti.Vector([0.1, 0.4, 0.8])
                # 使用 wave 在基礎藍色和更亮的藍色之間過渡，產生波紋
                color = base_blue * (0.8 + 0.2 * wave)
            else:  # 外部
                # 基礎顏色為橙色
                base_orange = ti.Vector([1.0, 0.5, 0.2])
                # 距離越遠，顏色越暗 (指數衰減)
                dist_attenuation = ti.exp(-sdf_val * 10.0)
                # 橙色波紋疊加衰減效果
                color = base_orange * (0.5 + 0.5 * wave) * dist_attenuation

            # 當 SDF 值非常接近 0 時，畫一條白色的邊界線
            if ti.abs(sdf_val) < 0.005:
                color = ti.Vector([1.0, 1.0, 1.0])

            # 將最終計算出的顏色賦值給畫布像素
            self.canvas[i, j] = color

    # --- 更新幾何數據的函式 (CPU) ---
    def update_geometry(self, polygons_list):
        edge_data = [] # 暫存邊緣的 Python 列表
        for poly in polygons_list:
            if len(poly) < 3:
                continue

            # 同樣需要確保多邊形是逆時針方向，以保證 Winding Number 計算的正確性
            area = 0.0
            for i in range(len(poly)):
                j = (i + 1) % len(poly)
                area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
            if area < 0:
                poly = poly[::-1] # 反轉

            # 建立邊緣列表
            for i in range(len(poly)):
                p1 = poly[i]
                p2 = poly[(i + 1) % len(poly)]
                edge_data.append([p1, p2])

        # 如果沒有邊緣數據，直接返回
        if not edge_data:
            # (可以選擇在這裡清空 sdf_field 和 canvas)
            self.num_edges[None] = 0
            self.sdf_field.fill(1e9) # 用一個很大的正數填充，表示全都是外部
            self.render_isolines()
            return

        # --- 數據傳輸到 GPU ---
        count = min(len(edge_data), self.MAX_EDGES) # 確保不超過最大邊緣數
        np_edges = np.zeros((self.MAX_EDGES, 2, 2), dtype=np.float32)
        np_edges[:count] = np.array(edge_data[:count], dtype=np.float32)

        self.edges.from_numpy(np_edges)
        self.num_edges[None] = count

        # --- 觸發 GPU 計算和渲染 ---
        # 依序呼叫核心函式
        self.compute_sdf_gpu()   # 第一步：計算 SDF
        self.render_isolines() # 第二步：根據 SDF 結果渲染視覺效果
