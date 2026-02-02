import taichi as ti
import numpy as np

# 注意：不要在這裡寫 ti.init()，它由 main.py 統一管理


@ti.data_oriented
class SDFEngine:
    """
    符號距離場計算引擎 (Signed Distance Field Engine)
    計算 2D 多邊形的 SDF 並產生視覺化的等高線圖。
    """

    def __init__(self, data_res=256, gui_res=512, max_edges=2000):
        self.res = data_res  # 計算網格解析度
        self.gui_res = gui_res  # GUI 顯示解析度
        self.MAX_EDGES = max_edges  # 最大邊緣數

        # Taichi Fields
        # 距離場
        self.sdf_field = ti.field(dtype=ti.f32, shape=(self.res, self.res))
        # 畫布 (RGB)
        self.canvas = ti.Vector.field(
            3, dtype=ti.f32, shape=(self.gui_res, self.gui_res)
        )

        # 邊緣數據儲存: shape=(邊數, 2個頂點) -> 每個元素是一個 Vector(2)
        self.edges = ti.Vector.field(2, dtype=ti.f32, shape=(self.MAX_EDGES, 2))
        self.num_edges = ti.field(dtype=ti.i32, shape=())

    @ti.func
    def dist_point_segment(self, p, v0, v1):
        """
        [GPU] 計算點到線段的最短距離。
        修正: 移除了 if 內的 return，改用變數賦值。
        """
        result = 0.0

        l2 = (v1 - v0).norm_sqr()

        # 使用一個很小的數 (epsilon) 來避免浮點數比較誤差
        if l2 < 1e-6:
            # 如果線段長度為 0 (起點=終點)，直接算點到點距離
            result = (p - v0).norm()
        else:
            # 投影 p 到線段 v0-v1 所在的直線上
            t = (p - v0).dot(v1 - v0) / l2
            t = ti.max(0.0, ti.min(1.0, t))  # 限制 t 在 [0, 1] 之間
            projection = v0 + t * (v1 - v0)
            result = (p - projection).norm()

        return result

    @ti.kernel
    def compute_sdf_gpu(self):
        """
        [GPU] 計算符號距離場 (SDF)。
        結合到最近邊緣的距離和環繞數來判斷內外。
        """
        for i, j in self.sdf_field:
            p = ti.Vector([i / self.res, j / self.res])

            min_dist = 1e9  # 初始化一個極大值
            winding = 0

            for k in range(self.num_edges[None]):
                v0 = self.edges[k, 0]
                v1 = self.edges[k, 1]

                # 1. 計算到最近邊緣的距離
                dist = self.dist_point_segment(p, v0, v1)
                min_dist = ti.min(min_dist, dist)

                # 2. 計算環繞數以判斷內外 (Point-in-Polygon)
                # 這裡不需要 return，所以 if 邏輯是安全的
                is_upward = (v0.y <= p.y) & (p.y < v1.y)
                is_downward = (v1.y <= p.y) & (p.y < v0.y)

                if is_upward or is_downward:
                    # 計算射線與邊的交點 X 座標
                    intersect_x = (v1.x - v0.x) * (p.y - v0.y) / (v1.y - v0.y) + v0.x
                    if p.x < intersect_x:
                        if is_upward:
                            winding += 1
                        else:
                            winding -= 1

            # 根據環繞數設定 SDF 值的符號 (內部為負，外部為正)
            if winding != 0:
                self.sdf_field[i, j] = -min_dist
            else:
                self.sdf_field[i, j] = min_dist

    @ti.kernel
    def render_isolines(self):
        """[GPU] 將 SDF 渲染為視覺化的等高線圖。"""
        for i, j in self.canvas:
            # 座標映射：從 canvas 解析度映射回 data 解析度
            data_i = i * self.res // self.gui_res
            data_j = j * self.res // self.gui_res

            sdf_val = self.sdf_field[data_i, data_j]

            # 使用 sin 函數產生波紋效果 (頻率 150.0)
            wave = ti.sin(sdf_val * 150.0)

            color = ti.Vector([0.0, 0.0, 0.0])

            # 根據內外選擇不同色調
            if sdf_val < 0:  # 內部
                # 藍色調 + 波紋
                base = ti.Vector([0.1, 0.4, 0.8])
                color = base * (0.6 + 0.4 * wave)
            else:  # 外部
                # 橘色調 + 距離衰減 + 波紋
                attenuation = ti.exp(-sdf_val * 15.0)  # 距離越遠越暗
                base = ti.Vector([1.0, 0.5, 0.2])
                color = base * (0.5 + 0.5 * wave) * attenuation

            # 繪製零等值線 (物體邊界)，白色
            if abs(sdf_val) < 0.005:
                color = ti.Vector([1.0, 1.0, 1.0])

            self.canvas[i, j] = color

    def update_geometry(self, polygons_list):
        """
        [CPU] 更新幾何數據，並觸發 GPU 計算與渲染。
        """
        edge_data = []
        for poly in polygons_list:
            if len(poly) < 3:
                continue

            # 計算多邊形面積 (Signed Area)
            area = 0.0
            for i in range(len(poly)):
                j = (i + 1) % len(poly)
                area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]

            # 如果是順時針 (面積 < 0)，反轉為逆時針
            # 確保 Union 運算正確
            if area < 0:
                poly = poly[::-1]

            # 構建邊緣列表
            for i in range(len(poly)):
                p1 = poly[i]
                p2 = poly[(i + 1) % len(poly)]
                edge_data.append([p1, p2])

        # 如果沒有數據，清空畫面
        if not edge_data:
            self.num_edges[None] = 0
            self.sdf_field.fill(100.0)  # 設為外部
            self.canvas.fill(0)
            return

        # 限制邊緣數量並轉換為 Numpy 格式
        count = min(len(edge_data), self.MAX_EDGES)
        np_edges = np.zeros((self.MAX_EDGES, 2, 2), dtype=np.float32)

        if count > 0:
            np_edges[:count] = np.array(edge_data[:count], dtype=np.float32)

        # 傳輸數據到 GPU
        self.edges.from_numpy(np_edges)
        self.num_edges[None] = count

        # 依序觸發 GPU 計算
        self.compute_sdf_gpu()
        self.render_isolines()
