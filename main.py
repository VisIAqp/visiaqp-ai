"""
main.py
Orquesta el motor de visiaqp-ai. Dos modos de ejecución (--modo, o la
variable VISIAQP_MODE; por defecto "camaras"):

  camaras  Procesa las cámaras en vivo (del backend, o cameras_fallback en dev).
  videos   Bucle de larga duración: sincroniza el catálogo de la carpeta de
           videos y procesa los procesamientos pendientes que pidan los usuarios.

Para correr ambos a la vez, lanzar dos procesos. Ver README.md para el ciclo
de vida del procesamiento y la configuración por entorno (VISIAQP_ENV).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.tracker import PedestrianTracker
from src.video_source import VideoSource
from src.video_library import VideoLibraryError, resolve_safe, scan
from src.zones import ZoneManager
from src.analyzer import ZoneAnalyzer
from src.notifier import Notifier
from src.visualizer import Visualizer


# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------

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
    """Carga config.base.yaml y le aplica los overrides de config.{env}.yaml."""
    base_path = Path(config_dir) / "config.base.yaml"
    env_path = Path(config_dir) / f"config.{env}.yaml"

    with open(base_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    with open(env_path, "r", encoding="utf-8") as f:
        overrides = yaml.safe_load(f) or {}

    return _deep_merge(base, overrides)


def load_cameras(config: dict, notifier: Notifier) -> list[dict]:
    """
    Cámaras a procesar: primero el backend (GET /api/camaras); si falla y
    fallback_to_local_cameras está activo, usa cameras_fallback.
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


# ---------------------------------------------------------------------
# Procesamiento de una fuente (cámara en vivo o video pregrabado)
# ---------------------------------------------------------------------

def _resolve_event_start(camera_config: dict, is_live: bool, nombre: str) -> datetime:
    """
    Hora de inicio del EVENTO:
      - fuente en vivo: ahora mismo.
      - archivo con `event_start`: esa fecha/hora (en videos pendientes es
        la fecha de captura que confirmó el usuario).
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


def _mensaje_error_publico(error: Exception, source, alias: str | None) -> str:
    """
    El mensaje de error viaja al frontend: para videos pregrabados se
    reemplaza la ruta absoluta del equipo del motor por el nombre del archivo.
    """
    mensaje = str(error)
    if alias:
        mensaje = mensaje.replace(str(source), alias)
    return mensaje


def _close_procesamiento(
    notifier: Notifier, id_procesamiento: int, estado: str, mensaje_error: str | None, log_cfg: dict,
) -> None:
    """Cierra el procesamiento en el backend; si falla, deja rastro en el fallback local."""
    cerrado = notifier.close_procesamiento(id_procesamiento, {
        "estado": estado,
        "fecha_fin_proceso": datetime.now().isoformat(),
        "mensaje_error": mensaje_error,
    })
    if not cerrado:
        _log_fallback("cierre_procesamiento", {
            "id_procesamiento": id_procesamiento,
            "estado": estado,
            "mensaje_error": mensaje_error,
        }, log_cfg)


def process_camera(
    camera_config: dict,
    config: dict,
    notifier: Notifier,
    id_procesamiento_existente: int | None = None,
) -> None:
    """
    Procesa UNA fuente de principio a fin: detección, tracking, filtro por
    zona, agregación por ventana, envío de resultados y cierre.

    id_procesamiento_existente: si viene (video pendiente ya reclamado), NO se
    crea un procesamiento nuevo: se usa ese id, con tipo_fuente="video".
    """
    id_camara = camera_config["id_camara"]
    id_zona = camera_config["id_zona"]
    source = camera_config["source"]
    nombre = camera_config.get("nombre", f"Camara {id_camara}")
    alias_archivo = camera_config.get("nombre_archivo")
    zone_polygon = camera_config.get("zone_polygon")
    es_video_pendiente = id_procesamiento_existente is not None

    if not zone_polygon:
        logger.error(f"{nombre}: no tiene zone_polygon definido, se omite")
        return

    logger.info(f"Iniciando procesamiento: {nombre} (id_camara={id_camara}, zona={id_zona})")

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

    # Si el procesamiento ya existe, se asigna ANTES del try: así, aunque el
    # video no se pueda abrir, el bloque finally lo cierra con estado "error".
    id_procesamiento = id_procesamiento_existente
    estado_final = "completado"
    mensaje_error = None

    try:
        with VideoSource(
            source,
            resize_width=processing_cfg["resize_width"],
            target_fps=processing_cfg["target_fps"],
        ) as video:
            event_start = _resolve_event_start(camera_config, video.is_live, nombre)
            tipo_fuente = "video" if es_video_pendiente else ("stream" if video.is_live else "video")

            logger.info(f"{nombre}: inicio del evento = {event_start.isoformat()} (tipo_fuente={tipo_fuente})")

            if not es_video_pendiente:
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

            last_event_time = event_start

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
                    display = visualizer.draw_info(display, len(en_zona), tracker.total_unique_count())
                    if not visualizer.show(display):
                        logger.info("Visualizador cerrado por el usuario (tecla 'q')")
                        break

                if analyzer.window_elapsed(event_time):
                    _close_window_and_send(analyzer, tracker, id_camara, id_procesamiento, notifier, log_cfg)

            if analyzer.has_pending_samples():
                logger.info(f"Cerrando ventana final parcial para {nombre}")
                _close_window_and_send(analyzer, tracker, id_camara, id_procesamiento, notifier, log_cfg)

    except KeyboardInterrupt:
        estado_final = "error"
        mensaje_error = "Procesamiento interrumpido manualmente"
        raise

    except Exception as e:
        logger.exception(f"No se pudo procesar {nombre}: {e}")
        estado_final = "error"
        mensaje_error = _mensaje_error_publico(e, source, alias_archivo if es_video_pendiente else None)

    finally:
        if visualizer:
            visualizer.close()

        if id_procesamiento is not None:
            _close_procesamiento(notifier, id_procesamiento, estado_final, mensaje_error, log_cfg)


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
    Si el notifier no pudo enviar el dato tras sus reintentos (o no había
    id_procesamiento para asociarlo), lo guarda localmente para no perderlo.
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


# ---------------------------------------------------------------------
# Modo "camaras"
# ---------------------------------------------------------------------

def run_cameras_mode(config: dict, notifier: Notifier) -> None:
    cameras = load_cameras(config, notifier)
    if not cameras:
        logger.warning("No hay cámaras para procesar")
        return

    for camera_config in cameras:
        process_camera(camera_config, config, notifier)


# ---------------------------------------------------------------------
# Modo "videos"
# ---------------------------------------------------------------------

def _sync_catalog(videos_cfg: dict, notifier: Notifier) -> None:
    """
    Escanea la carpeta y sincroniza el catálogo. Si la carpeta no es
    accesible NO se sincroniza: enviar una lista vacía haría que la API
    marcara todos los videos como no disponibles.
    """
    try:
        videos = scan(videos_cfg["library_dir"], videos_cfg["extensions"], videos_cfg.get("recursive", False))
    except VideoLibraryError as e:
        logger.error(f"{e}. Se omite la sincronización del catálogo.")
        return

    logger.info(f"Carpeta escaneada: {len(videos)} video(s)")
    if not notifier.sync_videos(videos):
        logger.warning("No se pudo sincronizar el catálogo; se reintenta en el próximo ciclo de sincronización")


def _handle_pending_video(pendiente: dict, config: dict, notifier: Notifier) -> bool:
    """
    Reclama y procesa un video pendiente. Devuelve True si el motor lo
    reclamó (y por tanto lo resolvió, bien o con error), False si lo omitió.
    """
    videos_cfg = config["videos"]
    log_cfg = config.get("logging", {})

    id_proc = pendiente.get("id_procesamiento")
    if id_proc is None:
        logger.warning(f"Pendiente sin id_procesamiento, se ignora: {pendiente}")
        return False

    nombre_archivo = pendiente.get("nombre_archivo")

    claim = notifier.claim_procesamiento(id_proc)
    if claim == "conflict":
        logger.info(f"Procesamiento {id_proc} ya fue tomado por otra instancia; se omite")
        return False
    if claim != "ok":
        logger.warning(f"No se pudo reclamar el procesamiento {id_proc}; se reintenta en el próximo ciclo")
        return False

    ruta = resolve_safe(videos_cfg["library_dir"], nombre_archivo, videos_cfg["extensions"])
    if ruta is None:
        logger.error(f"Procesamiento {id_proc}: archivo no encontrado o no permitido: {nombre_archivo!r}")
        _close_procesamiento(
            notifier, id_proc, "error",
            f"Archivo no encontrado o no permitido: {nombre_archivo}", log_cfg,
        )
        return True

    if not pendiente.get("zone_polygon"):
        logger.error(f"Procesamiento {id_proc}: la cámara no tiene zone_polygon")
        _close_procesamiento(notifier, id_proc, "error", "La cámara no tiene zone_polygon definido", log_cfg)
        return True

    camera_config = {
        "id_camara": pendiente["id_camara"],
        "id_zona": pendiente["id_zona"],
        "nombre": pendiente.get("nombre_camara") or f"Camara {pendiente['id_camara']}",
        "nombre_archivo": nombre_archivo,
        "source": str(ruta),
        "zone_polygon": pendiente["zone_polygon"],
        "event_start": pendiente.get("fecha_captura_inicio"),
    }
    process_camera(camera_config, config, notifier, id_procesamiento_existente=id_proc)
    return True


def run_videos_mode(config: dict, notifier: Notifier, max_cycles: int | None = None, sleep=time.sleep) -> None:
    """
    Bucle de larga duración: sincroniza el catálogo cada catalog_sync_seconds,
    consulta pendientes cada pending_poll_seconds y procesa de uno en uno
    (un video completo antes de volver a consultar). Si la API no responde
    no aborta: registra y espera al siguiente ciclo.

    max_cycles y sleep existen para poder probar el bucle.
    """
    videos_cfg = config.get("videos")
    if not videos_cfg:
        logger.error("Falta la sección 'videos' en la configuración")
        return

    sync_seconds = videos_cfg.get("catalog_sync_seconds", 60)
    poll_seconds = videos_cfg.get("pending_poll_seconds", 10)
    logger.info(
        f"Modo videos: carpeta={videos_cfg['library_dir']}, "
        f"sincronización cada {sync_seconds}s, consulta de pendientes cada {poll_seconds}s"
    )

    last_sync: float | None = None
    cycles = 0

    while max_cycles is None or cycles < max_cycles:
        cycles += 1

        if last_sync is None or time.monotonic() - last_sync >= sync_seconds:
            _sync_catalog(videos_cfg, notifier)
            last_sync = time.monotonic()

        pendientes = notifier.get_pending()
        trabajo_hecho = False

        if pendientes is None:
            logger.warning("No se pudo consultar los pendientes; se reintenta en el próximo ciclo")
        else:
            for pendiente in pendientes:
                if _handle_pending_video(pendiente, config, notifier):
                    trabajo_hecho = True
                    break  # un video a la vez: se vuelve a consultar antes del siguiente

        if not trabajo_hecho:
            sleep(poll_seconds)


# ---------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Motor de IA de VisIAqp")
    parser.add_argument(
        "--modo",
        default=os.environ.get("VISIAQP_MODE", "camaras"),
        help="camaras (en vivo) o videos (pregrabados). Por defecto: VISIAQP_MODE o 'camaras'",
    )
    args = parser.parse_args()
    if args.modo not in ("camaras", "videos"):
        parser.error(f"modo no válido: {args.modo!r} (usa 'camaras' o 'videos')")

    env = os.environ.get("VISIAQP_ENV", "dev")
    config = load_config(env)

    logging_cfg = config.get("logging", {})
    logger.remove()
    logger.add(sys.stderr, level=logging_cfg.get("level", "INFO"))

    logger.info(f"Entorno activo: {env} | modo: {args.modo}")

    notifier_cfg = config["notifier"]
    notifier = Notifier(
        api_base_url=os.environ.get("API_BASE_URL", "http://localhost:8000"),
        endpoints=notifier_cfg["endpoints"],
        api_key=os.environ.get("MOTOR_API_KEY") or os.environ.get("API_TOKEN") or None,
        timeout_seconds=notifier_cfg["timeout_seconds"],
        retry_attempts=notifier_cfg["retry_attempts"],
        retry_backoff_seconds=notifier_cfg["retry_backoff_seconds"],
    )

    try:
        if args.modo == "videos":
            run_videos_mode(config, notifier)
        else:
            run_cameras_mode(config, notifier)
    except KeyboardInterrupt:
        logger.info("Detenido por el usuario")


if __name__ == "__main__":
    main()