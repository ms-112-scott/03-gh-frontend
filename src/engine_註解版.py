# 匯入必要的函式庫
import taichi as ti  # 用於 GPU 計算
import numpy as np  # 用於數值計算，特別是與 Taichi 進行數據交換
import asyncio  # 用於非同步操作 (在此檔案的 run_loop 中使用)

# --- Taichi 資料導向類別 ---
# @ti.data_oriented 裝飾器告訴 Taichi 這是一個資料導向的類別
# 這使得類別中的 `ti.kernel` 和 `ti.func` 可以存取類別成員 (例如 self.mask_field)
@ti.data_oriented
class ComputeEngine:
    # --- 初始化函式 ---
    # 在建立 ComputeEngine 物件時會被呼叫
    def __init__(self, data_res=256, gui_res=512):
        # --- 成員變數定義 ---
        self.res = data_res  # 計算網格的解析度 (例如 256x256)
        self.gui_res = gui_res  # GUI 視窗的解析度 (例如 512x512)

        # --- Taichi Field 定義 ---
        # Field 是 Taichi 中用於儲存數據的容器，它會被分配在指定的計算後端 (例如 GPU) 的記憶體中
        # 建立一個 2D 的浮點數 field，用於儲存每個像素是否在多邊形內部的遮罩值 (0 或 1)
        self.mask_field = ti.field(dtype=ti.f32, shape=(self.res, self.res))
        # 建立一個 2D 的浮點數 field，用於最終在 GUI 上顯示的畫布
        self.canvas = ti.field(dtype=ti.f32, shape=(self.gui_res, self.gui_res))

        # --- 邊緣數據儲存 ---
        self.MAX_EDGES = 2000  # 設定一個靜態的最大邊緣數量，以預先分配記憶體
        # 建立一個 2D 向量 field 來儲存所有多邊形的邊緣
        # shape=(self.MAX_EDGES, 2) 表示有 MAX_EDGES 條邊，每條邊有 2 個頂點
        # ti.Vector.field(2, ...) 表示每個頂點是一個 2D 向量 (x, y)
        self.edges = ti.Vector.field(2, dtype=ti.f32, shape=(self.MAX_EDGES, 2))
        # 建立一個 0D 的整數 field 來儲存目前有效的邊緣數量
        # shape=() 表示這是一個純量 (scalar)
        self.num_edges = ti.field(dtype=ti.i32, shape=())

        # --- GUI 和狀態 ---
        # 建立一個獨立的 GUI 視窗 (雖然在主程式中統一管理，但此處展示了引擎也可以有自己的視窗)
        self.gui = ti.GUI("GPU SDF Union Mode", res=self.gui_res)
        self.is_running = True  # 控制渲染迴圈的狀態

    # --- Taichi GPU 核心函式 ---
    # @ti.kernel 裝飾器表示這個函式將會在 GPU 上並行執行
    @ti.kernel
    def compute_mask_gpu(self):
        """
        改進版：使用 Winding Number (環繞數) 算法在 GPU 上計算遮罩
        這個算法可以正確處理多個重疊多邊形的聯集 (Boolean Union)，
        避免了簡單奇偶規則在重疊處會產生空洞的問題。
        """
        # `for i, j in self.mask_field:` 是一個並行迴圈
        # Taichi 會自動將這個迴圈的每一次迭代分配給一個 GPU 核心去執行
        for i, j in self.mask_field:
            # 將像素的整數索引 (i, j) 轉換為標準化的浮點數座標 (p)，範圍在 [0, 1] 之間
            p = ti.Vector([i / self.res, j / self.res])

            # 初始化環繞數計數器為 0
            winding = 0

            # 遍歷目前所有的有效邊緣
            for e_idx in range(self.num_edges[None]):
                # 取得邊緣的兩個頂點 v0 和 v1
                # self.num_edges[None] 是讀取 0D field `num_edges` 的值
                v0 = self.edges[e_idx, 0]
                v1 = self.edges[e_idx, 1]

                # --- Winding Number 算法核心 ---
                # 從點 p 向右發射一條水平射線，計算這條射線與邊緣的相交情況
                # 判斷射線是否在邊緣的 Y 軸範圍內
                if v0.y <= p.y < v1.y:  # 邊緣是向上穿越射線 (Upward crossing)
                    # 計算射線與邊緣的交點的 X 座標
                    # 如果交點在點 p 的右邊 (即點 p 在交點的左邊)，表示射線與邊緣相交
                    if p.x < (v1.x - v0.x) * (p.y - v0.y) / (v1.y - v0.y) + v0.x:
                        winding += 1  # 環繞數加 1
                elif v1.y <= p.y < v0.y:  # 邊緣是向下穿越射線 (Downward crossing)
                    if p.x < (v1.x - v0.x) * (p.y - v0.y) / (v1.y - v0.y) + v0.x:
                        winding -= 1  # 環繞數減 1

            # 檢查最終的環繞數
            # 如果環繞數不為 0，表示點 p 在多邊形內部
            # 如果為 0，表示點 p 在外部
            self.mask_field[i, j] = 1.0 if winding != 0 else 0.0

    # --- Python CPU 函式 ---
    def _ensure_ccw(self, poly):
        """
        強制多邊形頂點為逆時針方向 (Counter-Clockwise, CCW)。
        原理：計算多邊形的帶符號面積 (Signed Area)。
        如果面積為負，表示頂點是順時針排列，則將頂點列表反轉。
        這對於 Winding Number 算法至關重要，因為它依賴於邊緣方向的一致性來實現聯集。
        """
        area = 0.0
        num_verts = len(poly)
        for i in range(num_verts):
            j = (i + 1) % num_verts  # 下一個頂點的索引
            area += poly[i][0] * poly[j][1]
            area -= poly[j][0] * poly[i][1]

        # 面積乘以 0.5 才是真實面積，但這裡我們只需要判斷正負
        if area < 0:
            return poly[::-1]  # 反轉列表
        return poly

    # --- 更新幾何數據的函式 ---
    # 這個函式在 CPU 上執行，準備好數據後傳輸到 GPU
    def update_geometry(self, polygons_list, bounds):
        edge_data = []  # 建立一個 Python 列表來暫存所有邊緣

        # 遍歷從客戶端傳來的所有多邊形
        for poly in polygons_list:
            if len(poly) < 3:
                continue  # 一個有效的多邊形至少需要 3 個頂點

            # 1. 強制所有多邊形都為逆時針方向 (關鍵步驟!)
            # 這樣可以確保當兩個多邊形重疊時，它們的環繞數是相加 (1+1=2)，而不是相互抵消 (1-1=0)。
            clean_poly = self._ensure_ccw(poly)

            # 2. 從多邊形的頂點構建邊緣列表
            num_verts = len(clean_poly)
            for i in range(num_verts):
                p1 = clean_poly[i]  # 當前頂點
                p2 = clean_poly[(i + 1) % num_verts]  # 下一個頂點 (最後一個頂點連回第一個)
                edge_data.append([p1, p2])  # 將這條邊加入列表

        # 如果沒有任何邊緣數據 (例如傳入空列表)，則清空畫布
        if not edge_data:
            self.num_edges[None] = 0
            self.mask_field.fill(0)  # 將遮罩 field 全部填為 0
            # 即使沒有邊緣，也需要呼叫一次 GPU 函式來更新畫布狀態
            self.compute_mask_gpu()
            return

        # 檢查邊緣數量是否超過預先分配的最大值
        current_count = len(edge_data)
        if current_count > self.MAX_EDGES:
            print(f"警告: 邊緣數量 {current_count} 超出最大值，將被截斷。")
            edge_data = edge_data[:self.MAX_EDGES]  # 截斷列表
            current_count = self.MAX_EDGES

        # 3. 準備數據並傳輸到 GPU
        # 建立一個符合 Taichi Field 形狀的 NumPy 陣列
        np_edges = np.zeros((self.MAX_EDGES, 2, 2), dtype=np.float32)
        # 將 Python 列表轉換為 NumPy 陣列並填入
        np_edges[:current_count] = np.array(edge_data, dtype=np.float32)

        # 使用 from_numpy() 將 CPU 上的 NumPy 陣列數據一次性傳輸到 GPU 上的 Taichi Field
        self.edges.from_numpy(np_edges)
        # 更新有效的邊緣數量
        self.num_edges[None] = current_count

        # 呼叫 GPU 核心函式，觸發計算
        self.compute_mask_gpu()

    # --- Taichi GPU 核心函式 ---
    @ti.kernel
    def _upsample(self):
        """
        升採樣函式：將低解析度的計算結果 (mask_field) 放大到高解析度的畫布 (canvas) 上。
        這是一種效能優化技巧：用較低解析度進行昂貴的計算，然後用較高解析度進行顯示。
        """
        # 並行遍歷高解析度的畫布
        for i, j in self.canvas:
            # 透過整數除法，找到 canvas 上的像素 (i, j) 對應到 mask_field 中的哪個像素
            # 這樣可以實現最近鄰插值 (Nearest-neighbor interpolation)
            self.canvas[i, j] = self.mask_field[
                i * self.res // self.gui_res, j * self.res // self.gui_res
            ]

    # --- 非同步渲染迴圈 (範例) ---
    async def run_loop(self):
        # 這個迴圈主要用於獨立測試此引擎
        while self.is_running and self.gui.running:
            self._upsample()  # 更新畫布
            self.gui.set_image(self.canvas)  # 將畫布設定到 GUI
            self.gui.show()  # 顯示
            await asyncio.sleep(0.01)  # 非同步等待
