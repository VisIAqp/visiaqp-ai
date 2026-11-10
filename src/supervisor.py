"""Supervisor local con procesos independientes y concurrencia limitada."""
import copy
import multiprocessing
import time

from loguru import logger


def trabajador_camara(camara, config, notifier, parada):
    # Importación local para que Windows pueda crear procesos mediante spawn.
    from main import process_camera
    from src.video_source import VideoSource

    en_vivo = VideoSource.es_fuente_en_vivo(camara['source'])
    espera = config.get('multicamara', {}).get('reintento_seg', 5)
    while not parada.is_set():
        estado = process_camera(camara, config, notifier, stop_event=parada)
        if estado == 'reiniciar_proceso':
            raise SystemExit(1)
        if not en_vivo or parada.is_set():
            return
        logger.warning(f"Cámara {camara['id_camara']}: {estado}; se reintentará en {espera}s")
        parada.wait(espera)


def supervisar(config, notifier, cargar, contexto=None, max_cycles=None):
    contexto = contexto or multiprocessing.get_context('spawn')
    ajustes = config.get('multicamara', {})
    limite = int(ajustes.get('max_camaras', 2))
    if limite < 1:
        raise ValueError('max_camaras debe ser mayor que cero')
    intervalo = float(ajustes.get('actualizar_seg', 10))
    if intervalo <= 0:
        raise ValueError('actualizar_seg debe ser mayor que cero')
    procesos = {}
    terminadas = {}
    pendientes_cierre = {}
    ultimo_catalogo = {}
    ciclos = 0
    # Las ventanas de OpenCV se reservan para ejecuciones individuales de depuración.
    config_hijo = copy.deepcopy(config)
    config_hijo.setdefault('visualizer', {})['enabled'] = False

    def detener(id_camara):
        proceso, parada, _ = procesos.pop(id_camara)
        parada.set()
        pendientes_cierre[id_camara] = (proceso, time.monotonic())

    try:
        while max_cycles is None or ciclos < max_cycles:
            ciclos += 1
            nuevas = cargar(config, notifier)
            # Un fallo del backend no equivale a eliminar todas las cámaras.
            if nuevas is not None:
                ultimo_catalogo = {c['id_camara']: c for c in nuevas if c.get('zone_polygon')}
            for id_camara, (proceso, parada, anterior) in list(procesos.items()):
                if id_camara not in ultimo_catalogo or anterior != ultimo_catalogo[id_camara]:
                    detener(id_camara)
                    terminadas.pop(id_camara, None)
                elif not proceso.is_alive():
                    proceso.join()
                    del procesos[id_camara]
                    if proceso.exitcode == 0:
                        terminadas[id_camara] = anterior
            for id_camara, (proceso, inicio_cierre) in list(pendientes_cierre.items()):
                if proceso.is_alive() and time.monotonic() - inicio_cierre > ajustes.get('cierre_seg', 20):
                    proceso.terminate()
                    proceso.join(timeout=2)
                if not proceso.is_alive():
                    proceso.join()
                    del pendientes_cierre[id_camara]
            for id_camara, camara in ultimo_catalogo.items():
                if len(procesos) + len(pendientes_cierre) >= limite:
                    break
                if id_camara in procesos or id_camara in pendientes_cierre or terminadas.get(id_camara) == camara:
                    continue
                parada = contexto.Event()
                proceso = contexto.Process(target=trabajador_camara, args=(camara, config_hijo, notifier, parada))
                proceso.start()
                procesos[id_camara] = (proceso, parada, copy.deepcopy(camara))
                logger.info(f'Cámara {id_camara}: proceso independiente iniciado')
            if max_cycles is None or ciclos < max_cycles:
                time.sleep(intervalo)
    finally:
        for id_camara in list(procesos):
            detener(id_camara)
        limite_cierre = time.monotonic() + ajustes.get('cierre_seg', 20)
        for proceso, _ in pendientes_cierre.values():
            proceso.join(timeout=max(0, limite_cierre - time.monotonic()))
            if proceso.is_alive():
                proceso.terminate()
                proceso.join(timeout=2)
