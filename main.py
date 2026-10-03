"""
main.py
Orquesta el pipeline completo de visiaqp-ai: lee config.yaml y .env,
arma los componentes (tracker, zonas, analyzer, notifier) por cada
cámara configurada, y corre el loop principal de procesamiento.

Tiempo de evento vs. tiempo de registro:
  - Los timestamps que se envían a la API son de EVENTO (cuándo ocurrió
    lo que muestra el video): hora de inicio + segundos transcurridos
    en la fuente. No dependen de qué tan rápido procese la CPU.
  - El momento en que la API guarda el dato (fecha_registro) lo pone la
    base de datos con su DEFAULT CURRENT_TIMESTAMP.

Pendiente conocido:
  - TODO: cargar las cámaras desde la base de datos en vez de config.yaml.
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


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Carga y parsea config.yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_event_start(camera_config: dict, is_live: bool, nombre: str) -> datetime:
    """
    Determina la hora de inicio del EVENTO para esta cámara:
      - fuente en vivo: ahora mismo.
      - archivo con `event_start` en el config: esa fecha/hora.
      - archivo sin `event_start`: se asume ahora (con advertencia), porque
        no hay forma confiable de saber cuándo se grabó.
    """
    if is_live:
        return datetime.now()

    raw = camera_config.get("event_start")
    if raw is None:
        logger.warning(
            f"{nombre}: archivo de video sin 'event_start' en config.yaml; "
            f"se asume que el evento ocurre ahora. Defínelo para registrar la fecha real."
        )
        return datetime.now()

    # PyYAML convierte fechas sin comillas a datetime automáticamente
    if isinstance(raw, datetime):
        return raw
    return datetime.fromisoformat(str(raw))


def process_camera(camera_config: dict, config: dict, notifier: Notifier) -> None:
    """
    Procesa UNA cámara de principio a fin: abre su fuente de video,
    corre detección+tracking, filtra por zona, agrega por ventana,
    y notifica cada ventana cerrada al backend.
    """
    id_camara = camera_config["id_camara"]
    id_zona = camera_config["id_zona"]
    source = camera_config["source"]
    nombre = camera_config.get("nombre", f"Camara {id_camara}")

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

    analyzer = ZoneAnalyzer(
        id_zona=id_zona,
        window_seconds=analysis_cfg["window_seconds"],
        thresholds=analysis_cfg["thresholds"],
    )

    # El visualizer solo se crea si está habilitado (debug local, nunca en producción)
    visualizer = None
    if vis_cfg.get("enabled", False):
        visualizer = Visualizer(
            window_name=vis_cfg.get("window_name", "VisIAqp - Debug"),
            show_boxes=vis_cfg.get("show_boxes", True),
        )

    try:
        with VideoSource(
            source,
            resize_width=processing_cfg["resize_width"],
            target_fps=processing_cfg["target_fps"],
        ) as video:
            event_start = _resolve_event_start(camera_config, video.is_live, nombre)
            tipo_fuente = "video"
            last_event_time = event_start

            logger.info(f"{nombre}: inicio del evento = {event_start.isoformat()}")

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
                    _close_and_notify(
                        analyzer, tracker, id_camara, tipo_fuente, event_time, notifier, log_cfg
                    )

            # Al terminar el video (o cortarse el stream), cerrar la
            # ventana parcial en curso si tiene muestras pendientes,
            # en vez de descartarla silenciosamente.
            if analyzer.has_pending_samples():
                logger.info(f"Cerrando ventana final parcial para {nombre}")
                _close_and_notify(
                    analyzer, tracker, id_camara, tipo_fuente, last_event_time, notifier, log_cfg
                )

    except RuntimeError as e:
        logger.error(f"No se pudo procesar la cámara {nombre}: {e}")

    finally:
        if visualizer:
            visualizer.close()


def _close_and_notify(
    analyzer: ZoneAnalyzer,
    tracker: PedestrianTracker,
    id_camara: int,
    tipo_fuente: str,
    event_time: datetime,
    notifier: Notifier,
    log_cfg: dict,
) -> None:
    """
    Cierra la ventana actual y notifica al backend dos cosas:
      - el resultado puntual (tabla resultado_deteccion)
      - el indicador de concentración (tabla indicador)

    IMPORTANTE: build_resultado() debe llamarse ANTES de close_window(),
    porque close_window() limpia las muestras que build_resultado() usa
    para calcular la confianza promedio.
    """
    unique_in_window = tracker.window_unique_count()

    resultado = analyzer.build_resultado(
        id_camara=id_camara,
        tipo_fuente=tipo_fuente,
        cantidad_peatones_unicos=unique_in_window,
        timestamp=event_time,
    )
    indicador = analyzer.close_window(unique_in_window)
    tracker.reset_window()

    if not notifier.send_resultado(resultado):
        _log_fallback("resultado", resultado.to_dict(), log_cfg)

    if indicador is not None and not notifier.send_indicador(indicador):
        _log_fallback("indicador", indicador.to_dict(), log_cfg)


def _log_fallback(kind: str, payload: dict, log_cfg: dict) -> None:
    """
    Si notifier.py no pudo enviar el dato tras sus reintentos, lo guarda
    localmente (si fallback_to_local_log está activo) para no perderlo.
    Cada línea del .jsonl indica el tipo ("resultado" | "indicador"), para
    poder reenviarlo después al endpoint correcto.
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
    config = load_config()

    logging_cfg = config.get("logging", {})
    logger.remove()
    logger.add(sys.stderr, level=logging_cfg.get("level", "INFO"))

    notifier_cfg = config["notifier"]
    notifier = Notifier(
        api_base_url=os.environ.get("API_BASE_URL", "http://localhost:8000"),
        endpoints=notifier_cfg["endpoints"],
        api_token=os.environ.get("API_TOKEN") or None,
        timeout_seconds=notifier_cfg["timeout_seconds"],
        retry_attempts=notifier_cfg["retry_attempts"],
        retry_backoff_seconds=notifier_cfg["retry_backoff_seconds"],
    )

    cameras = config.get("cameras", [])
    if not cameras:
        logger.warning("No hay cámaras configuradas en config.yaml")
        return

    for camera_config in cameras:
        process_camera(camera_config, config, notifier)


if __name__ == "__main__":
    main()