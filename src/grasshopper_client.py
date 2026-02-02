# run in grasshopper ghpython script component
# r: websocket-client
import websocket
import scriptcontext as sc
import json
import Rhino.Geometry as rg


# --- 幾何數據預處理 (不變) ---
def get_payload(tree, bbox_rect):
    if not bbox_rect:
        return None
    bbox = bbox_rect.GetBoundingBox(True)
    min_p, max_p = bbox.Min, bbox.Max
    width, height = max_p.X - min_p.X, max_p.Y - min_p.Y

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

            current_poly_pts = [
                [(pt.X - min_p.X) / width, (pt.Y - min_p.Y) / height] for pt in polyline
            ]
            all_polygons.append(current_poly_pts)

    return json.dumps({"polygons": all_polygons, "bounds": {"aspect": width / height}})


# --- 同步連線與發送邏輯 ---

url = "ws://localhost:8765"

if run:
    try:
        # 1. 取得或建立連線
        ws = sc.sticky.get("sync_ws")

        # 檢查連線是否還活著，如果關閉了就重連
        if ws is None or not ws.connected:
            # 使用完整路徑呼叫
            ws = websocket.create_connection(url, timeout=0.5)
            sc.sticky["sync_ws"] = ws
            status = "Connected (Sync)"
        else:
            status = "Sending..."

        # 2. 準備並發送
        payload_str = get_payload(polyline_tree, boundary)
        if payload_str:
            ws.send(payload_str)

    except Exception as e:
        status = f"Connect Error: {e}"
        # 出錯時清除連線，下次重試
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
    status = "Off"

a = f"Status: {status}"
