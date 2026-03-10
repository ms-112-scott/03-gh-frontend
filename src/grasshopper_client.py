# grasshopper_client.py  (VAENCA 整合版)
# Run in Grasshopper GHPython component
#
# ── 輸入端口 ─────────────────────────────────────────────────────────────────
#   run          : bool      - True 啟動連線
#   polyline_tree: DataTree  - 多邊形幾何
#   boundary     : Rectangle - 邊界矩形
#   mode         : str       - "sdf" | "mask" | "nca"  (預設 "nca")
#   viz_mode     : str       - "velocity" | "vorticity" | "pressure"
#                              "beaufort" | "stress"    (預設 "velocity")
#   img_save_path: str       - JPEG 儲存路徑（選填）
#
# ── 輸出端口 ─────────────────────────────────────────────────────────────────
#   a : str       - 狀態訊息（含模型類型、速度、Beaufort、VAENCA 受體權重）
#   b : dict      - 速度場統計 + receptor_weights（VAENCA 限定）
#   c : list[dict]- 蒲氏級數圖例

import websocket
import scriptcontext as sc
import json
import Rhino.Geometry as rg
import System
import System.Drawing
import System.IO


# ── 幾何預處理 ────────────────────────────────────────────────────────────────

def get_payload(tree, bbox_rect, send_mode, send_viz_mode):
    """把 GH 幾何轉成正規化座標 JSON。"""
    if not bbox_rect:
        return None
    bbox   = bbox_rect.GetBoundingBox(True)
    min_p  = bbox.Min
    max_p  = bbox.Max
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
            pts = [
                [(pt.X - min_p.X) / width, (pt.Y - min_p.Y) / height]
                for pt in polyline
            ]
            all_polygons.append(pts)

    return json.dumps({
        "polygons": all_polygons,
        "bounds":   {"aspect": width / height},
        "mode":     send_mode,
        "viz_mode": send_viz_mode,
    })


# ── 格式化輔助 ────────────────────────────────────────────────────────────────

def format_legend(legend_list):
    """把圖例 list 轉為 GH Panel 可顯示的文字。"""
    lines = ["蒲氏風力級數色階："]
    for item in legend_list:
        max_str = f"< {item['max_ms']:.1f} m/s" if item["max_ms"] else "≥ 32.7 m/s"
        lines.append(f"  {item['color_hex']}  {item['label']}  {max_str}")
    return "\n".join(lines)


def format_receptor_weights(rw: dict) -> str:
    """把 VAENCA receptor_weights 格式化為單行文字。"""
    if not rw:
        return ""
    return (
        f"Receptors → "
        f"A(拓撲):{rw.get('A_topo', 0):.3f}  "
        f"B(平流):{rw.get('B_stream', 0):.3f}  "
        f"C(MRT):{rw.get('C_mrt', 0):.3f}  "
        f"D(學習):{rw.get('D_learned', 0):.3f}  "
        f"E(荷爾蒙):{rw.get('E_hormone', 0):.3f}"
    )


# ── 主邏輯 ────────────────────────────────────────────────────────────────────

url       = "ws://localhost:8765"
send_mode = mode     if "mode"     in dir() and mode     else "nca"
send_viz  = viz_mode if "viz_mode" in dir() and viz_mode else "velocity"
save_path = img_save_path if "img_save_path" in dir() else ""

status_str = "Off"
vel_stats  = {}
legend_out = []

if run:
    try:
        # ── 取得或重建 WebSocket 連線 ─────────────────────────────────────
        ws = sc.sticky.get("sync_ws")
        if ws is None or not ws.connected:
            ws = websocket.create_connection(url, timeout=1.0)
            sc.sticky["sync_ws"] = ws
            status_str = "Connected"
        else:
            status_str = "Sending..."

        # ── 組建並發送 payload ───────────────────────────────────────────
        payload_str = get_payload(polyline_tree, boundary, send_mode, send_viz)

        if payload_str is None:
            # 無幾何時只送 viz_mode 切換訊號
            ws.send(json.dumps({"viz_mode": send_viz}))
        else:
            ws.send(payload_str)

        # ── 接收回應 ────────────────────────────────────────────────────
        raw  = ws.recv()
        resp = json.loads(raw)

        if resp.get("status") == "ok":
            vm    = resp.get("viz_mode", send_viz)
            stats = resp.get("stats", {})
            vel_stats = stats

            mtype = stats.get("model_type", "nca")
            bft   = stats.get("beaufort_max", "?")
            spd   = stats.get("speed_max_ms", 0.0)

            # 基本狀態行
            status_str = f"OK | {mtype.upper()} | {vm} | {spd:.1f} m/s | Bft {bft}"

            # VAENCA 受體權重行
            rw = stats.get("receptor_weights", {})
            if rw:
                status_str += "\n" + format_receptor_weights(rw)

            # 蒲氏圖例
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
                bmp        = System.Drawing.Bitmap(img_stream)
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
            except Exception:
                pass
            del sc.sticky["sync_ws"]

else:
    if "sync_ws" in sc.sticky:
        try:
            sc.sticky["sync_ws"].close()
        except Exception:
            pass
        del sc.sticky["sync_ws"]
    status_str = "Off"
    legend_out = sc.sticky.get("beaufort_legend", [])

# ── 輸出 ──────────────────────────────────────────────────────────────────────
a = f"Status: {status_str}"
b = vel_stats       # dict，含 speed_max_ms, beaufort_max, receptor_weights 等
c = legend_out      # list[dict]，蒲氏圖例
