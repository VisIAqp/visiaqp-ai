"""
main.py
Orquesta el pipeline de visiaqp-ai: obtiene cámaras del backend (o del
fallback local), procesa cada una (detección, tracking, zonas, ventanas)
y envía los resultados. Ver README.md para el ciclo de vida completo del
procesamiento y la configuración por entorno (VISIAQP_ENV).
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.tracker import PedestrianTracker
from src.video_source import VideoSource
from src.zones import ZoneManager
from src.analyzer import ZoneAnalyzer
from src.notifier import Notifier
from src.visualizer import Visualizer


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Fusiona overrides sobre base, recursivamente para diccionarios anidados."""
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(env: str = "dev", config_dir: str = "config") -> dict:
    """
    Carga config.base.yaml y le aplica los overrides de config.{env}.yaml.
    env: "dev" (por defecto) o "prod".
    """
    base_path = Path(config_dir) / "config.base.yaml"
    env_path = Path(config_dir) / f"config.{env}.yaml"

    with open(base_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}

    with open(env_path, "r", encoding="utf-8") as f:
        overrides = yaml.safe_load(f) or {}

    return _deep_merge(base, overrides)


def load_cameras(config: dict, notifier: Notifier) -> list[dict]:
    """
    Obtiene la lista de cámaras a procesar: primero intenta el backend
    (GET /api/camaras); si falla y fallback_to_local_cameras está
    activo, usa cameras_fallback de config.{env}.yaml.
    """
    cameras = notifier.get_cameras()
    if cameras is not None:
        logger.info(f"{len(cameras)} cámara(s) obtenidas del backend")
        return cameras

    if config.get("fallback_to_local_cameras", False):
        local = config.get("cameras_fallback", [])
        logger.warning(
            f"No se pudo obtener la lista de cámaras del backend; "
            f"usando cameras_fallback de config ({len(local)} cámara(s))"
        )
        return local

    logger.error("No se pudo obtener la lista de cámaras del backend y no hay fallback local configurado")
    return []


def _resolve_event_start(camera_config: dict, is_live: bool, nombre: str) -> datetime:
    """
    Determina la hora de inicio del EVENTO para esta cámara:
      - fuente en vivo: ahora mismo.
      - archivo con `event_start` en su configuración: esa fecha/hora.
      - archivo sin `event_start`: se asume ahora (con advertencia).
    """
    if is_live:
        return datetime.now()

    raw = camera_config.get("event_start")
    if raw is None:
        logger.warning(
            f"{nombre}: archivo de video sin 'event_start'; "
            f"se asume que el evento ocurre ahora. Defínelo para registrar la fecha real."
        )
        return datetime.now()

    if isinstance(raw, datetime):
        return raw
    return datetime.fromisoformat(str(raw))


def process_camera(camera_config: dict, config: dict, notifier: Notifier) -> None:
    """
    Procesa UNA cámara de principio a fin: crea el procesamiento en el
    backend, abre la fuente de video, corre detección+tracking, filtra
    por zona, agrega por ventana, envía cada resultado, y cierra el
    procesamiento al terminar (éxito o error).
    """
    id_camara = camera_config["id_camara"]
    id_zona = camera_config["id_zona"]
    source = camera_config["source"]
    nombre = camera_config.get("nombre", f"Camara {id_camara}")
    zone_polygon = camera_config.get("zone_polygon")

    if not zone_polygon:
        logger.error(f"{nombre}: no tiene zone_polygon definido, se omite")
        return

    logger.info(f"Iniciando procesamiento de cámara: {nombre} (id={id_camara}, zona={id_zona})")

    model_cfg = config["model"]
    tracker_cfg = config["tracker"]
    processing_cfg = config["processing"]
    analysis_cfg = config["analysis"]
    log_cfg = config.get("logging", {})
    vis_cfg = config.get("visualizer", {})

    tracker = PedestrianTracker(
        weights_path=model_cfg["weights_path"],
        device=model_cfg["device"],
        confidence_threshold=model_cfg["confidence_threshold"],
        iou_threshold=model_cfg["iou_threshold"],
        classes=model_cfg["classes"],
        algorithm=tracker_cfg["algorithm"],
        track_thresh=tracker_cfg["track_thresh"],
        track_buffer=tracker_cfg["track_buffer"],
        match_thresh=tracker_cfg["match_thresh"],
    )

    zone_manager = ZoneManager.from_config(camera_config)
    analyzer = ZoneAnalyzer(id_zona=id_zona, window_seconds=analysis_cfg["window_seconds"])

    visualizer = None
    if vis_cfg.get("enabled", False):
        visualizer = Visualizer(
            window_name=vis_cfg.get("window_name", "VisIAqp - Debug"),
            show_boxes=vis_cfg.get("show_boxes", True),
        )

    id_procesamiento = None
    estado_final = "completado"
    mensaje_error = None

    try:
        with VideoSource(
            source,
            resize_width=processing_cfg["resize_width"],
            target_fps=processing_cfg["target_fps"],
        ) as video:
            event_start = _resolve_event_start(camera_config, video.is_live, nombre)
            tipo_fuente = "stream" if video.is_live else "video"
            last_event_time = event_start

            logger.info(f"{nombre}: inicio del evento = {event_start.isoformat()} (tipo_fuente={tipo_fuente})")

            procesamiento_payload = {
                "id_zona": id_zona,
                "id_camara": id_camara,
                "tipo_fuente": tipo_fuente,
                "ruta_archivo": str(source) if not video.is_live else None,
                "fecha_captura_inicio": event_start.isoformat(),
            }
            id_procesamiento = notifier.create_procesamiento(procesamiento_payload)
            if id_procesamiento is None:
                logger.error(
                    f"{nombre}: no se pudo crear el procesamiento en el backend; "
                    f"se continúa solo con fallback local"
                )
                _log_fallback("creacion_procesamiento", procesamiento_payload, log_cfg)

            for frame, elapsed_seconds in video.frames():
                event_time = event_start + timedelta(seconds=elapsed_seconds)
                last_event_time = event_time

                tracked = tracker.track(frame)
                h, w = frame.shape[:2]
                en_zona = zone_manager.filter_detections_in_zone(tracked, w, h)

                analyzer.add_sample(en_zona, event_time)

                if visualizer:
                    display = visualizer.draw_zone(frame, zone_manager.zones[0], w, h)
                    display = visualizer.draw_detections(display, en_zona)
                    display = visualizer.draw_info(
                        display, len(en_zona), tracker.total_unique_count()
                    )
                    if not visualizer.show(display):
                        logger.info("Visualizador cerrado por el usuario (tecla 'q')")
                        break

                if analyzer.window_elapsed(event_time):
                    _close_window_and_send(analyzer, tracker, id_camara, id_procesamiento, notifier, log_cfg)

            if analyzer.has_pending_samples():
                logger.info(f"Cerrando ventana final parcial para {nombre}")
                _close_window_and_send(analyzer, tracker, id_camara, id_procesamiento, notifier, log_cfg)

    except RuntimeError as e:
        logger.error(f"No se pudo procesar la cámara {nombre}: {e}")
        estado_final = "error"
        mensaje_error = str(e)

    finally:
        if visualizer:
            visualizer.close()

        if id_procesamiento is not None:
            cerrado = notifier.close_procesamiento(id_procesamiento, {
                "estado": estado_final,
                "fecha_fin_proceso": datetime.now().isoformat(),
                "mensaje_error": mensaje_error,
            })
            if not cerrado:
                _log_fallback("cierre_procesamiento", {
                    "id_procesamiento": id_procesamiento,
                    "estado": estado_final,
                    "mensaje_error": mensaje_error,
                }, log_cfg)


def _close_window_and_send(
    analyzer: ZoneAnalyzer,
    tracker: PedestrianTracker,
    id_camara: int,
    id_procesamiento: int | None,
    notifier: Notifier,
    log_cfg: dict,
) -> None:
    """Cierra la ventana actual y envía el ResultadoDeteccion crudo resultante."""
    unique_in_window = tracker.window_unique_count()
    resultado = analyzer.close_window(id_camara, unique_in_window)
    tracker.reset_window()

    if resultado is None:
        return

    if id_procesamiento is None or not notifier.send_resultado(id_procesamiento, resultado):
        _log_fallback("resultado", resultado.to_dict(), log_cfg)


def _log_fallback(kind: str, payload: dict, log_cfg: dict) -> None:
    """
    Si notifier.py no pudo enviar el dato tras sus reintentos (o si no
    había id_procesamiento para asociarlo), lo guarda localmente (si
    fallback_to_local_log está activo) para no perderlo.
    """
    if not log_cfg.get("fallback_to_local_log", True):
        logger.error(f"Falló el envío de {kind} y el fallback local está desactivado: dato perdido")
        return

    log_dir = Path(log_cfg.get("log_dir", "data/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"fallback_{datetime.now().strftime('%Y%m%d')}.jsonl"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"tipo": kind, "payload": payload}, ensure_ascii=False) + "\n")

    logger.warning(f"{kind.capitalize()} guardado localmente en {log_file} (falló el envío a la API)")


def main() -> None:
    load_dotenv()

    env = os.environ.get("VISIAQP_ENV", "dev")
    config = load_config(env)

    logging_cfg = config.get("logging", {})
    logger.remove()
    logger.add(sys.stderr, level=logging_cfg.get("level", "INFO"))

    logger.info(f"Entorno activo: {env}")

    notifier_cfg = config["notifier"]
    notifier = Notifier(
        api_base_url=os.environ.get("API_BASE_URL", "http://localhost:8000"),
        endpoints=notifier_cfg["endpoints"],
        api_token=os.environ.get("API_TOKEN") or None,
        timeout_seconds=notifier_cfg["timeout_seconds"],
        retry_attempts=notifier_cfg["retry_attempts"],
        retry_backoff_seconds=notifier_cfg["retry_backoff_seconds"],
    )

    cameras = load_cameras(config, notifier)
    if not cameras:
        logger.warning("No hay cámaras para procesar")
        return

    for camera_config in cameras:
        process_camera(camera_config, config, notifier)


if __name__ == "__main__":
    main()