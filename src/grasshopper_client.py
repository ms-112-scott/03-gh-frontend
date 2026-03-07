# grasshopper_client.py  (NCA 版)
# Run in Grasshopper GHPython component
#
# 輸入腳本端口（在 GH 元件上設定）：
#   run          : bool      - True 啟動連線
#   polyline_tree: DataTree  - 多邊形幾何
#   boundary     : Rectangle - 邊界矩形
#   mode         : str       - "sdf" | "mask" | "nca"  (預設 "nca")
#   img_save_path: str       - NCA 影像儲存路徑（選填，留空則不儲存）
#
# 輸出端口：
#   a      : 狀態字串
#   b      : NCA 速度場統計字典（僅 NCA 模式有值）

import websocket
import scriptcontext as sc
import json
import Rhino.Geometry as rg
import System
import System.Drawing


# ── 幾何預處理（與原版相同）─────────────────────────────────────────────
def get_payload(tree, bbox_rect, send_mode):
    if not bbox_rect:
        return None
    bbox  = bbox_rect.GetBoundingBox(True)
    min_p, max_p = bbox.Min, bbox.Max
    width  = max_p.X - min_p.X
    height = max_p.Y - min_p.Y
    if width <= 0 or height <= 0:
        return None

    all_polygons = []
    for i in range(tree.BranchCount):
        branch = tree.Branch(i)
        for poly_geo in branch:
            success, polyline = poly_geo.TryGetPolyline()
            if not success:
                temp_pc = poly_geo.ToPolyline(0.1, 0, 0.1, 0)
                if temp_pc:
                    polyline = temp_pc.ToPolyline()
                else:
                    continue

            pts = [
                [(pt.X - min_p.X) / width, (pt.Y - min_p.Y) / height]
                for pt in polyline
            ]
            all_polygons.append(pts)

    return json.dumps({
        "polygons": all_polygons,
        "bounds": {"aspect": width / height},
        "mode": send_mode,          # 每次發送都帶上模式，確保伺服器同步
    })


# ── 主邏輯 ────────────────────────────────────────────────────────────────
url        = "ws://localhost:8765"
send_mode  = mode if "mode" in dir() and mode else "nca"   # 預設 NCA 模式
save_path  = img_save_path if "img_save_path" in dir() else ""

status_str = "Off"
vel_stats  = {}

if run:
    try:
        # ── 取得或建立 WebSocket 連線 ──────────────────────────────────
        ws = sc.sticky.get("sync_ws")
        if ws is None or not ws.connected:
            ws = websocket.create_connection(url, timeout=1.0)
            sc.sticky["sync_ws"] = ws
            status_str = "Connected"
        else:
            status_str = "Sending..."

        # ── 準備並發送 payload ─────────────────────────────────────────
        payload_str = get_payload(polyline_tree, boundary, send_mode)
        if payload_str:
            ws.send(payload_str)

            # ── 接收回應 ───────────────────────────────────────────────
            raw = ws.recv()
            resp = json.loads(raw)

            if resp.get("status") == "ok":
                current_mode = resp.get("mode", "?")
                status_str = f"OK | mode={current_mode}"

                # 速度場統計（NCA 模式）
                if "stats" in resp:
                    vel_stats = resp["stats"]
                    status_str += f" | speed_max={vel_stats.get('speed_max', 0):.4f}"

                # 速度場 JPEG 影像（若伺服器有附帶）
                if "jpeg_b64" in resp and resp["jpeg_b64"]:
                    import base64
                    img_bytes = base64.b64decode(resp["jpeg_b64"])
                    img_stream = System.IO.MemoryStream(img_bytes)
                    bmp = System.Drawing.Bitmap(img_stream)

                    # 儲存至磁碟（選填）
                    if save_path:
                        bmp.Save(save_path)
                        status_str += f" | img saved"

            elif resp.get("status") == "error":
                status_str = f"Server Error: {resp.get('msg', '')}"

    except Exception as e:
        status_str = f"Error: {e}"
        # 連線失敗時清除，下次重連
        if "sync_ws" in sc.sticky:
            try:
                sc.sticky["sync_ws"].close()
            except:
                pass
            del sc.sticky["sync_ws"]

else:
    # run = False：關閉連線
    if "sync_ws" in sc.sticky:
        try:
            sc.sticky["sync_ws"].close()
        except:
            pass
        del sc.sticky["sync_ws"]
    status_str = "Off"

# ── 輸出 ──────────────────────────────────────────────────────────────────
a = f"Status: {status_str}"
b = vel_stats
