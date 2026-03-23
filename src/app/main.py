from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import taichi as ti
import websockets
import yaml

from src.integration import NCAEngine, PhysicalConditions, load_inference_pipeline
from src.middleware.mask_engine import ComputeEngine
from src.middleware.sdf_engine import SDFEngine


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

ti.init(arch=ti.gpu)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("RhinoBridge")

engine_config = {
    "data_res": config["data_res"],
    "gui_res": config["gui_res"],
    "max_edges": config["max_edges"],
}
engine_mask = ComputeEngine(**engine_config)
engine_sdf = SDFEngine(**engine_config)

engine_nca: NCAEngine | None = None
pipeline_info: dict[str, Any] | None = None
try:
    pipeline_info = load_inference_pipeline(config)
    engine_nca = NCAEngine(
        pipeline=pipeline_info["pipeline"],
        data_res=config["data_res"],
        gui_res=config["gui_res"],
        viz_mode=config.get("default_viz_mode", "velocity"),
    )
    logger.info(
        "[Main] inference adapter ready | backend_root=%s",
        pipeline_info["backend_root"],
    )
except Exception as exc:
    logger.error("[Main] failed to load inference pipeline: %s", exc, exc_info=True)

gui = ti.GUI("Rhino NCA Bridge", res=config["gui_res"])

current_mode = "sdf"
latest_polygons: list = []
nca_steps_per_frame = config.get("nca_steps_per_frame", 4)
current_conditions = PhysicalConditions.from_mapping(config.get("default_conditions"))

VIZ_MODE_LABELS = {
    "velocity": "Velocity",
    "vorticity": "Vorticity",
    "pressure": "Pressure",
    "beaufort": "Beaufort",
    "stress": "Stress",
}


def _refresh_geometry() -> None:
    if not latest_polygons:
        return

    engine_mask.update_geometry(latest_polygons, {})
    engine_sdf.update_geometry(latest_polygons)

    if current_mode == "nca" and engine_nca is not None:
        engine_nca.set_geometry(
            mask_np=engine_mask.mask_field.to_numpy(),
            sdf_np=engine_sdf.sdf_field.to_numpy(),
            conditions=current_conditions,
            cold=True,
        )


def _handle_conditions_update(payload: dict[str, Any]) -> None:
    global current_conditions
    current_conditions = PhysicalConditions.from_mapping(payload)
    if engine_nca is not None:
        engine_nca.update_conditions(current_conditions)


def _build_response(need_geo_update: bool) -> dict[str, Any]:
    response: dict[str, Any] = {"status": "ok", "mode": current_mode}

    if current_mode == "nca" and engine_nca is not None and engine_nca.latest_output is not None:
        latest_output = engine_nca.latest_output
        response["meta"] = latest_output.get("meta", {})
        response["stats"] = engine_nca.get_velocity_stats()
        response["viz_mode"] = engine_nca.viz_mode
        if need_geo_update:
            response["beaufort_legend"] = engine_nca.beaufort_legend()

    return response


async def ws_handler(websocket):
    global current_mode, latest_polygons
    logger.info("websocket client connected")

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                need_geo_update = False

                if "polygons" in data:
                    latest_polygons = data["polygons"]
                    need_geo_update = True

                if "mode" in data:
                    new_mode = data["mode"]
                    if new_mode == "nca" and engine_nca is None:
                        await websocket.send(
                            json.dumps(
                                {
                                    "status": "error",
                                    "msg": "Inference pipeline is not available.",
                                    "mode": current_mode,
                                }
                            )
                        )
                        continue
                    if new_mode != current_mode:
                        current_mode = new_mode
                        need_geo_update = True

                if "viz_mode" in data and engine_nca is not None:
                    engine_nca.set_viz_mode(data["viz_mode"])

                if "conditions" in data:
                    _handle_conditions_update(data["conditions"])

                if "dcnca_restart" in data and engine_nca is not None:
                    restart_flag = data["dcnca_restart"]
                    cold = restart_flag == "cold"
                    if restart_flag is True or cold:
                        engine_nca.restart(cold=cold)

                if "module_toggle" in data and engine_nca is not None:
                    for name, enabled in data["module_toggle"].items():
                        engine_nca.toggle_module(name, bool(enabled))

                if "output_toggle" in data and engine_nca is not None:
                    output_toggle = data["output_toggle"]
                    engine_nca.set_output_toggles(
                        output_moments=output_toggle.get("moments"),
                        output_phy_fields=output_toggle.get("phy_fields"),
                    )

                if need_geo_update:
                    if current_mode == "sdf":
                        engine_sdf.update_geometry(latest_polygons)
                    elif current_mode == "mask":
                        engine_mask.update_geometry(latest_polygons, {})
                    elif current_mode == "nca":
                        _refresh_geometry()

                if (
                    "conditions" in data
                    and not need_geo_update
                    and current_mode == "nca"
                    and engine_nca is not None
                ):
                    engine_nca.update_conditions(current_conditions)

                await websocket.send(json.dumps(_build_response(need_geo_update)))
            except Exception as exc:
                logger.error("[WS] request handling failed: %s", exc, exc_info=True)
    except websockets.exceptions.ConnectionClosed:
        logger.info("websocket client disconnected")


async def render_loop():
    while gui.running:
        if current_mode == "sdf":
            gui.set_image(engine_sdf.canvas)
            gui.text("Mode: SDF", pos=(0.02, 0.97), color=0xFFFFFF, font_size=18)
        elif current_mode == "mask":
            gui.set_image(engine_mask.canvas)
            gui.text("Mode: Mask", pos=(0.02, 0.97), color=0xFFFFFF, font_size=18)
        elif current_mode == "nca":
            if engine_nca is not None:
                engine_nca.step(nca_steps_per_frame)
                gui.set_image(engine_nca.canvas)
                stats = engine_nca.get_velocity_stats()
                viz_label = VIZ_MODE_LABELS.get(engine_nca.viz_mode, engine_nca.viz_mode)
                gui.text(
                    f"{viz_label} | {stats.get('speed_max_ms', 0.0):.2f} m/s | "
                    f"Bft {stats.get('beaufort_max', 0)}",
                    pos=(0.02, 0.97),
                    color=0xFFFF00,
                    font_size=18,
                )
            else:
                gui.set_image(engine_sdf.canvas)
                gui.text(
                    "NCA unavailable",
                    pos=(0.02, 0.97),
                    color=0xFF8800,
                    font_size=18,
                )

        gui.show()
        await asyncio.sleep(0.01)


async def main():
    ws_host = config["ws_host"]
    ws_port = config["ws_port"]
    await websockets.serve(ws_handler, ws_host, ws_port)
    logger.info("websocket server started at ws://%s:%s", ws_host, ws_port)
    logger.info("available viz_mode: %s", NCAEngine.available_viz_modes())
    await render_loop()


def run() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
