import queue
import threading

import numpy as np
import pytest

import main
from src.video_source import VideoSource, ErrorTransmision
from src.supervisor import supervisar


class Captura:
    def __init__(self):
        self.cola = queue.Queue()
        self.abierta = True

    def isOpened(self):
        return self.abierta

    def get(self, _):
        return 30

    def read(self):
        try:
            valor = self.cola.get(timeout=1)
        except queue.Empty:
            return False, None
        return valor is not None, valor

    def release(self):
        self.abierta = False


def imagen(valor):
    return np.full((10, 10, 3), valor, dtype=np.uint8)


def test_captura_en_vivo_descarta_frames_pendientes_y_no_confunde_caida_con_fin(monkeypatch):
    captura = Captura()
    monkeypatch.setattr('src.video_source.cv2.VideoCapture', lambda *a: captura)
    fuente = VideoSource(0, target_fps=None)
    for valor in (1, 2, 3):
        captura.cola.put(imagen(valor))
    with fuente._condicion:
        assert fuente._condicion.wait_for(lambda: fuente._secuencia == 3, timeout=1)
    generador = fuente.frames()
    frame, tiempo = next(generador)
    assert frame[0, 0, 0] == 3
    assert tiempo >= 0
    captura.cola.put(None)
    with fuente._condicion:
        assert fuente._condicion.wait_for(lambda: fuente._error, timeout=1)
    with pytest.raises(ErrorTransmision):
        next(generador)
    assert not captura.abierta


def test_archivo_usa_tiempo_de_contenido_y_termina_sin_error(monkeypatch):
    captura = Captura()
    for valor in range(12):
        captura.cola.put(imagen(valor))
    captura.cola.put(None)
    monkeypatch.setattr('src.video_source.cv2.VideoCapture', lambda *a: captura)
    with VideoSource('video.mp4', target_fps=5) as fuente:
        lecturas = list(fuente.frames())
    assert [t for _, t in lecturas] == pytest.approx([0.2, 0.4])
    assert not captura.abierta


class Proceso:
    def __init__(self, target, args):
        self.args = args
        self.exitcode = 0
        self.vivo = False

    def start(self):
        self.vivo = True

    def is_alive(self):
        return self.vivo and not self.args[-1].is_set()

    def join(self, timeout=None):
        pass

    def terminate(self):
        self.vivo = False


class Contexto:
    def __init__(self):
        self.creados = []

    Event = staticmethod(threading.Event)

    def Process(self, target, args):
        p = Proceso(target, args)
        self.creados.append(p)
        return p


def camara(id):
    return {'id_camara': id, 'id_zona': 1, 'source': f'rtsp://ejemplo/{id}',
            'zone_polygon': [[0, 0], [1, 0], [1, 1]]}


def test_supervisor_limita_concurrencia_y_aisla_eventos(monkeypatch):
    monkeypatch.setattr('src.supervisor.time.sleep', lambda _: None)
    contexto = Contexto()
    config = {'multicamara': {'max_camaras': 2}, 'visualizer': {'enabled': True}}
    supervisar(config, None, lambda *a: [camara(i) for i in range(4)], contexto=contexto, max_cycles=2)
    assert len(contexto.creados) == 2
    assert contexto.creados[0].args[-1] is not contexto.creados[1].args[-1]
    assert contexto.creados[0].args[1]['visualizer']['enabled'] is False
    assert config['visualizer']['enabled'] is True


def test_fallo_de_catalogo_no_detiene_camaras_y_cambio_reinicia_solo_la_afectada(monkeypatch):
    monkeypatch.setattr('src.supervisor.time.sleep', lambda _: None)
    contexto = Contexto()
    catalogos = iter([[camara(1), camara(2)], None,
                      [{**camara(1), 'intervalo_muestreo_seg': 30}, camara(2)]])
    supervisar({'multicamara': {'max_camaras': 2}}, None, lambda *a: next(catalogos),
               contexto=contexto, max_cycles=3)
    assert [p.args[0]['id_camara'] for p in contexto.creados] == [1, 2, 1]


def test_reserva_rechazada_no_carga_modelo_ni_abre_fuente(monkeypatch):
    class Notificador:
        def create_procesamiento(self, payload):
            return None
    def prohibido(*a, **k):
        pytest.fail('No debe cargarse el modelo sin reserva')
    monkeypatch.setattr(main, 'PedestrianTracker', prohibido)
    assert main.process_camera(camara(1), {}, Notificador()) == 'reserva_no_confirmada'


def test_error_al_cargar_modelo_cierra_procesamiento_reclamado(monkeypatch):
    class Notificador:
        def __init__(self):
            self.cierres = []
        def close_procesamiento(self, id, payload):
            self.cierres.append((id, payload))
            return True
    def fallar(**kwargs):
        raise RuntimeError('Modelo no disponible')
    monkeypatch.setattr(main, 'PedestrianTracker', fallar)
    notificador = Notificador()
    config = {'model': dict(weights_path='modelo', device='cpu', confidence_threshold=0.4,
                           iou_threshold=0.45, classes=[0]),
              'tracker': dict(algorithm='bytetrack', track_thresh=0.5, track_buffer=30, match_thresh=0.8),
              'processing': {}}
    assert main.process_camera(camara(1), config, notificador, 9) == 'error'
    assert notificador.cierres[0][0] == 9
    assert notificador.cierres[0][1]['estado'] == 'error'


def test_trabajador_reconecta_transmision_y_no_repite_archivo(monkeypatch):
    from src.supervisor import trabajador_camara
    parada = threading.Event()
    llamadas = []
    def procesar(*a, **k):
        llamadas.append(1)
        if len(llamadas) == 2:
            parada.set()
        return 'error'
    monkeypatch.setattr(main, 'process_camera', procesar)
    trabajador_camara(camara(1), {'multicamara': {'reintento_seg': 0}}, None, parada)
    assert len(llamadas) == 2
    parada.clear()
    llamadas.clear()
    trabajador_camara({**camara(1), 'source': 'video.mp4'}, {}, None, parada)
    assert len(llamadas) == 1


def test_fallo_de_lectura_en_vivo_cierra_con_error(monkeypatch):
    captura = Captura()
    captura.cola.put(None)
    monkeypatch.setattr('src.video_source.cv2.VideoCapture', lambda *a: captura)
    monkeypatch.setattr(main, 'PedestrianTracker', lambda **k: object())
    class Notificador:
        def __init__(self):
            self.cierres = []
        def create_procesamiento(self, payload):
            return 7
        def send_actividad(self, *a):
            return True
        def close_procesamiento(self, id, payload):
            self.cierres.append(payload)
            return True
    notificador = Notificador()
    config = main.load_config('dev')
    config['visualizer']['enabled'] = False
    assert main.process_camera({**camara(1), 'source': 0}, config, notificador) == 'error'
    assert notificador.cierres[0]['estado'] == 'error'
    assert 'transmisión' in notificador.cierres[0]['mensaje_error']


def test_captura_que_no_cierra_exige_reinicio_de_proceso():
    from src.video_source import CapturaBloqueada
    class Lector:
        def join(self, timeout):
            pass
        def is_alive(self):
            return True
    fuente = object.__new__(VideoSource)
    fuente._bloqueada = False
    fuente._parar = threading.Event()
    fuente._lector = Lector()
    fuente.read_timeout_seconds = 0
    with pytest.raises(CapturaBloqueada):
        fuente.release()


def test_trabajador_sale_si_captura_no_responde_para_que_supervisor_lo_recree(monkeypatch):
    from src.supervisor import trabajador_camara
    monkeypatch.setattr(main, 'process_camera', lambda *a, **k: 'reiniciar_proceso')
    with pytest.raises(SystemExit) as error:
        trabajador_camara(camara(1), {}, None, threading.Event())
    assert error.value.code == 1
