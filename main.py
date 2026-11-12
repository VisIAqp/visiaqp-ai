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
from src.video_source import VideoSource, CapturaBloqueada, ahora_lima
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


def load_cameras(config: dict, notifier: Notifier) -> list[dict] | None:
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
    return None


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
        return ahora_lima()

    raw = camera_config.get("event_start")
    if raw is None:
        logger.warning(
            f"{nombre}: archivo de video sin 'event_start'; "
            f"se asume que el evento ocurre ahora. Defínelo para registrar la fecha real."
        )
        return ahora_lima()

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
        "fecha_fin_proceso": ahora_lima().isoformat(),
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
    stop_event=None,
) -> str:
    """Una cámara, una reserva y un estado de tracking independientes."""
    import threading

    stop_event = stop_event or threading.Event()
    id_camara = camera_config['id_camara']
    id_zona = camera_config['id_zona']
    source = camera_config['source']
    nombre = camera_config.get('nombre', f'Cámara {id_camara}')
    log_cfg = config.get('logging', {})
    id_procesamiento = id_procesamiento_existente
    tracker = visualizer = analyzer = None
    estado_final, mensaje_error = 'completado', None
    reiniciar_proceso = False
    en_vivo = VideoSource.es_fuente_en_vivo(source)
    inicio = time.monotonic()
    frames = 0
    ultima_actividad = inicio
    ultima_muestra = None
    fin_bloque = None
    try:
        if not camera_config.get('zone_polygon'):
            raise ValueError('La cámara necesita un polígono de análisis')
        # Reservar antes de cargar YOLO o abrir la transmisión. Si el backend
        # no confirma la reserva, no se procesa ni se publica nada en paralelo.
        if id_procesamiento is None:
            id_procesamiento = notifier.create_procesamiento({
                'id_zona': id_zona, 'id_camara': id_camara,
                'tipo_fuente': 'stream' if en_vivo else 'video',
                'ruta_archivo': None if en_vivo else str(source),
                'fecha_captura_inicio': _resolve_event_start(camera_config, en_vivo, nombre).isoformat(),
                **({'firma_captura': camera_config['firma_captura']}
                   if en_vivo and camera_config.get('firma_captura') else {}),
            })
            if id_procesamiento is None:
                logger.warning(f'{nombre}: reserva no confirmada; se omite esta ejecución')
                return 'reserva_no_confirmada'
        model_cfg, tracker_cfg = config['model'], config['tracker']
        processing_cfg = config['processing']
        if str(model_cfg['device']).startswith('cpu'):
            import torch
            torch.set_num_threads(processing_cfg.get('cpu_threads', 2))
        tracker = PedestrianTracker(
            weights_path=model_cfg['weights_path'], device=model_cfg['device'],
            confidence_threshold=model_cfg['confidence_threshold'],
            iou_threshold=model_cfg['iou_threshold'], classes=model_cfg['classes'],
            algorithm=tracker_cfg['algorithm'], track_thresh=tracker_cfg['track_thresh'],
            track_buffer=tracker_cfg['track_buffer'], match_thresh=tracker_cfg['match_thresh'],
        )
        zone_manager = ZoneManager.from_config(camera_config)
        # El intervalo por cámara define la ventana de reporte; target_fps
        # controla por separado la frecuencia de análisis de imágenes.
        ventana = camera_config.get('intervalo_muestreo_seg', config['analysis']['window_seconds'])
        analyzer = ZoneAnalyzer(id_zona=id_zona, window_seconds=ventana)
        vis_cfg = config.get('visualizer', {})
        if vis_cfg.get('enabled', False):
            visualizer = Visualizer(window_name=vis_cfg.get('window_name', 'VisIAqp - Debug'),
                                    show_boxes=vis_cfg.get('show_boxes', True))
        if not notifier.send_actividad(id_procesamiento, {}):
            raise RuntimeError('No se pudo confirmar la vigencia de la reserva')
        with VideoSource(source, resize_width=processing_cfg['resize_width'],
                         target_fps=processing_cfg['target_fps'],
                         read_timeout_seconds=processing_cfg.get('read_timeout_seconds', 10),
                         stop_event=stop_event) as video:
            event_start = video.started_at if en_vivo else _resolve_event_start(camera_config, False, nombre)
            inicio = time.monotonic()
            for frame, elapsed_seconds in video.frames():
                if stop_event.is_set():
                    break
                event_time = event_start + timedelta(seconds=elapsed_seconds)
                # No interpolar ocupación a través de una interrupción prolongada.
                if en_vivo and ultima_muestra and (event_time - ultima_muestra).total_seconds() > processing_cfg.get('max_gap_seconds', 5):
                    _close_window_and_send(analyzer, tracker, id_camara, id_procesamiento, notifier, log_cfg)
                    tracker.reset()
                if en_vivo and (fin_bloque is None or event_time >= fin_bloque):
                    if analyzer.has_pending_samples():
                        _close_window_and_send(analyzer, tracker, id_camara, id_procesamiento, notifier, log_cfg)
                    inicio_bloque = event_time.replace(minute=event_time.minute // 5 * 5, second=0, microsecond=0)
                    fin_bloque = inicio_bloque + timedelta(minutes=5)
                tracked = tracker.track(frame)
                h, w = frame.shape[:2]
                en_zona = zone_manager.filter_detections_in_zone(tracked, w, h)
                analyzer.add_sample(en_zona, event_time)
                ultima_muestra = event_time
                frames += 1
                ahora = time.monotonic()
                if ahora - ultima_actividad >= config.get('multicamara', {}).get('actividad_seg', 10):
                    datos = {
                        'ultima_muestra': event_time.isoformat(),
                        'fps_procesamiento': frames / max(ahora - inicio, 0.001),
                        'retraso_seg': max(0, ahora - video.last_captured_monotonic) if en_vivo else 0,
                    }
                    if not notifier.send_actividad(id_procesamiento, datos):
                        raise RuntimeError('Se perdió la comunicación o la reserva; se detiene la fuente')
                    ultima_actividad = time.monotonic()
                if visualizer:
                    display = visualizer.draw_zone(frame, zone_manager.zones[0], w, h)
                    display = visualizer.draw_detections(display, en_zona)
                    display = visualizer.draw_info(display, len(en_zona), tracker.total_unique_count())
                    if not visualizer.show(display):
                        stop_event.set()
                        break
                if analyzer.window_elapsed(event_time):
                    _close_window_and_send(analyzer, tracker, id_camara, id_procesamiento, notifier, log_cfg)
            if stop_event.is_set():
                estado_final, mensaje_error = 'error', 'Procesamiento detenido por el supervisor'
    except KeyboardInterrupt:
        estado_final, mensaje_error = 'error', 'Procesamiento interrumpido manualmente'
        raise
    except Exception as error:
        reiniciar_proceso = isinstance(error, CapturaBloqueada)
        estado_final = 'error'
        mensaje_error = _mensaje_error_publico(error, source, camera_config.get('nombre_archivo') or 'fuente configurada')
        logger.error(f'{nombre}: {mensaje_error}')
    finally:
        if analyzer and tracker and analyzer.has_pending_samples():
            try:
                _close_window_and_send(analyzer, tracker, id_camara, id_procesamiento, notifier, log_cfg)
            except Exception:
                logger.exception(f'{nombre}: no se pudo cerrar la ventana parcial')
        if visualizer:
            visualizer.close()
        if id_procesamiento is not None:
            _close_procesamiento(notifier, id_procesamiento, estado_final, mensaje_error, log_cfg)
    return 'reiniciar_proceso' if reiniciar_proceso else estado_final


def _close_window_and_send(
    analyzer: ZoneAnalyzer,
    tracker: PedestrianTracker,
    id_camara: int,
    id_procesamiento: int | None,
    notifier: Notifier,
    log_cfg: dict,
) -> None:
    """Cierra la ventana actual y envía el ResultadoDeteccion crudo resultante."""
    resultado = analyzer.close_window(id_camara)
    tracker.reset_window()

    if resultado is None:
        return

    if id_procesamiento is None or not notifier.send_resultado(id_procesamiento, resultado):
        _log_fallback("resultado", {"id_procesamiento": id_procesamiento, **resultado.to_dict()}, log_cfg)


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
    log_file = log_dir / f"fallback_{datetime.now().strftime('%Y%m%d')}_{os.getpid()}.jsonl"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"tipo": kind, "payload": payload}, ensure_ascii=False) + "\n")

    logger.warning(f"{kind.capitalize()} guardado localmente en {log_file} (falló el envío a la API)")


# ---------------------------------------------------------------------
# Modo "camaras"
# ---------------------------------------------------------------------

def run_cameras_mode(config: dict, notifier: Notifier) -> None:
    from src.supervisor import supervisar
    supervisar(config, notifier, load_cameras)


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
        "intervalo_muestreo_seg": pendiente.get("intervalo_muestreo_seg", config.get("analysis", {}).get("window_seconds", 60)),
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
