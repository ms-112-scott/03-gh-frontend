# grasshopper_client.py  (視覺化切換版)
# Run in Grasshopper GHPython component
#
# ── 輸入端口 ─────────────────────────────────────────────────────────────
#   run          : bool      - True 啟動連線
#   polyline_tree: DataTree  - 多邊形幾何
#   boundary     : Rectangle - 邊界矩形
#   mode         : str       - "sdf" | "mask" | "nca"  (預設 "nca")
#   viz_mode     : str       - "velocity" | "vorticity" | "pressure"
#                              "beaufort" | "stress"    (預設 "beaufort")
#   img_save_path: str       - JPEG 儲存路徑（選填）
#
# ── 輸出端口 ─────────────────────────────────────────────────────────────
#   a            : str       - 狀態訊息
#   b            : dict      - 速度場統計 {speed_max_ms, beaufort_max, ...}
#   c            : list[dict]- 蒲氏級數圖例（初次連線時回傳）

import websocket
import scriptcontext as sc
import json
import Rhino.Geometry as rg
import System
import System.Drawing
import System.IO


# ── 幾何預處理 ────────────────────────────────────────────────────────────
def get_payload(tree, bbox_rect, send_mode, send_viz_mode):
    if not bbox_rect:
        return None
    bbox   = bbox_rect.GetBoundingBox(True)
    min_p  = bbox.Min;  max_p = bbox.Max
    width  = max_p.X - min_p.X
    height = max_p.Y - min_p.Y
    if width <= 0 or height <= 0:
        return None

    all_polygons = []
    for i in range(tree.BranchCount):
        for poly_geo in tree.Branch(i):
            ok, polyline = poly_geo.TryGetPolyline()
            if not ok:
                pc = poly_geo.ToPolyline(0.1, 0, 0.1, 0)
                polyline = pc.ToPolyline() if pc else None
            if polyline is None:
                continue
            pts = [[(pt.X - min_p.X)/width, (pt.Y - min_p.Y)/height]
                   for pt in polyline]
            all_polygons.append(pts)

    return json.dumps({
        "polygons": all_polygons,
        "bounds":   {"aspect": width / height},
        "mode":     send_mode,
        "viz_mode": send_viz_mode,
    })


# ── 蒲氏級數圖例轉為 GH 可顯示的格式 ────────────────────────────────────
def format_legend(legend_list):
    """把 legend list 轉成文字色表格（GH Panel 顯示用）"""
    lines = ["蒲氏風力級數色階："]
    for item in legend_list:
        max_str = f"< {item['max_ms']:.1f} m/s" if item["max_ms"] else "≥ 32.7 m/s"
        lines.append(f"  {item['color_hex']}  {item['label']}  {max_str}")
    return "\n".join(lines)


# ── 主邏輯 ────────────────────────────────────────────────────────────────
url          = "ws://localhost:8765"
send_mode    = mode     if "mode"     in dir() and mode     else "nca"
send_viz     = viz_mode if "viz_mode" in dir() and viz_mode else "beaufort"
save_path    = img_save_path if "img_save_path" in dir() else ""

status_str   = "Off"
vel_stats    = {}
legend_out   = []

if run:
    try:
        # ── 取得或重建連線 ────────────────────────────────────────────
        ws = sc.sticky.get("sync_ws")
        if ws is None or not ws.connected:
            ws = websocket.create_connection(url, timeout=1.0)
            sc.sticky["sync_ws"] = ws
            status_str = "Connected"
        else:
            status_str = "Sending..."

        # ── 純切換 viz_mode（無幾何更新）─────────────────────────────
        # 每次 GH 元件重算都會觸發，因此先判斷是否只有 viz_mode 改變
        payload_str = get_payload(polyline_tree, boundary, send_mode, send_viz)

        if payload_str is None:
            # 沒有幾何，只送 viz_mode 切換訊號
            ws.send(json.dumps({"viz_mode": send_viz}))
        else:
            ws.send(payload_str)

        # ── 接收回應 ──────────────────────────────────────────────────
        raw  = ws.recv()
        resp = json.loads(raw)

        if resp.get("status") == "ok":
            vm   = resp.get("viz_mode", send_viz)
            stats = resp.get("stats", {})
            vel_stats = stats

            bft  = stats.get("beaufort_max", "?")
            spd  = stats.get("speed_max_ms", 0.0)
            status_str = (
                f"OK | {vm} | {spd:.1f} m/s | Bft {bft}"
            )

            # 首次取得圖例
            if "beaufort_legend" in resp:
                legend_out = resp["beaufort_legend"]
                sc.sticky["beaufort_legend"] = legend_out
            elif "beaufort_legend" in sc.sticky:
                legend_out = sc.sticky["beaufort_legend"]

            # JPEG 處理
            if "jpeg_b64" in resp and resp["jpeg_b64"]:
                import base64
                img_bytes  = base64.b64decode(resp["jpeg_b64"])
                img_stream = System.IO.MemoryStream(img_bytes)
                bmp = System.Drawing.Bitmap(img_stream)
                if save_path:
                    bmp.Save(save_path)
                    status_str += " | img saved"

        elif resp.get("status") == "error":
            status_str = f"Server Error: {resp.get('msg', '')}"

    except Exception as e:
        status_str = f"Error: {e}"
        if "sync_ws" in sc.sticky:
            try:
                sc.sticky["sync_ws"].close()
            except:
                pass
            del sc.sticky["sync_ws"]

else:
    if "sync_ws" in sc.sticky:
        try:
            sc.sticky["sync_ws"].close()
        except:
            pass
        del sc.sticky["sync_ws"]
    status_str = "Off"
    legend_out = sc.sticky.get("beaufort_legend", [])

# ── 輸出 ──────────────────────────────────────────────────────────────────
a = f"Status: {status_str}"
b = vel_stats
c = legend_out
