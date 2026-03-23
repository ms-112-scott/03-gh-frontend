import taichi as ti
import numpy as np
import asyncio

@ti.data_oriented
class ComputeEngine:
    """
    二元遮罩計算引擎 (Binary Mask Engine)
    使用環繞數演算法 (Winding Number Algorithm) 產生多邊形的 2D 遮罩。
    """
    def __init__(self, data_res, gui_res, max_edges):
        self.res = data_res  # 計算網格解析度
        self.gui_res = gui_res  # GUI 顯示解析度
        self.MAX_EDGES = max_edges # 最大邊緣數

        # Taichi Fields
        self.mask_field = ti.field(dtype=ti.f32, shape=(self.res, self.res))
        self.canvas = ti.field(dtype=ti.f32, shape=(self.gui_res, self.gui_res))

        # 邊緣數據儲存
        self.edges = ti.Vector.field(2, dtype=ti.f32, shape=(self.MAX_EDGES, 2))
        self.num_edges = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def compute_mask_gpu(self):
        """
        [GPU] 使用環繞數演算法計算遮罩。
        此方法能正確處理重疊多邊形的聯集。
        """
        for i, j in self.mask_field:
            p = ti.Vector([i / self.res, j / self.res])
            winding = 0

            # 遍歷所有邊緣，計算環繞數
            for e_idx in range(self.num_edges[None]):
                v0, v1 = self.edges[e_idx, 0], self.edges[e_idx, 1]
                
                # 判斷水平射線是否與邊緣相交
                is_upward = v0.y <= p.y < v1.y
                is_downward = v1.y <= p.y < v0.y
                
                if is_upward or is_downward:
                    # 計算交點 x 座標
                    t = (p.y - v0.y) / (v1.y - v0.y)
                    intersect_x = v0.x + t * (v1.x - v0.x)
                    
                    if p.x < intersect_x:
                        if is_upward:
                            winding += 1
                        else:
                            winding -= 1

            # 環繞數不為 0 表示點在內部
            self.mask_field[i, j] = 1.0 if winding != 0 else 0.0

    def _ensure_ccw(self, poly):
        """
        [CPU] 確保多邊形頂點為逆時針 (CCW) 順序。
        對環繞數演算法至關重要。
        """
        area = 0.0
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
        
        if area < 0:
            return poly[::-1]  # 順時針則反轉
        return poly

    def update_geometry(self, polygons_list, bounds):
        """
        [CPU] 更新幾何數據，並觸發 GPU 計算。
        """
        edge_data = []
        for poly in polygons_list:
            if len(poly) < 3:
                continue

            # 1. 確保多邊形方向一致
            clean_poly = self._ensure_ccw(poly)

            # 2. 建立邊緣列表
            for i in range(len(clean_poly)):
                p1 = clean_poly[i]
                p2 = clean_poly[(i + 1) % len(clean_poly)]
                edge_data.append([p1, p2])

        if not edge_data:
            self.num_edges[None] = 0
            self.mask_field.fill(0)
            self._upsample() # 更新畫布
            return

        # 3. 準備數據並傳輸至 GPU
        count = min(len(edge_data), self.MAX_EDGES)
        np_edges = np.zeros((self.MAX_EDGES, 2, 2), dtype=np.float32)
        if count > 0:
            np_edges[:count] = np.array(edge_data[:count], dtype=np.float32)

        self.edges.from_numpy(np_edges)
        self.num_edges[None] = count

        # 4. 觸發 GPU 計算與畫布更新
        self.compute_mask_gpu()
        self._upsample()

    @ti.kernel
    def _upsample(self):
        """
        [GPU] 將低解析度的計算結果升採樣至高解析度的畫布。
        """
        for i, j in self.canvas:
            self.canvas[i, j] = self.mask_field[
                i * self.res // self.gui_res, j * self.res // self.gui_res
            ]
