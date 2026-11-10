"""Medición local de inferencia; no escribe resultados en la API."""
import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path


def medir_camara(indice, video, pesos, frames, hilos, barrera, salida):
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import cv2
        import psutil
        import torch
        from src.tracker import PedestrianTracker

        torch.set_num_threads(hilos)
        modelo = PedestrianTracker(pesos, device='cpu')
        captura = cv2.VideoCapture(video)
        if not captura.isOpened():
            raise RuntimeError('No se pudo abrir el video de medición')
        fps = captura.get(cv2.CAP_PROP_FPS)
        salto = max(1, round(fps / 5))
        def siguiente():
            imagen = None
            for _ in range(salto):
                ok, imagen = captura.read()
                if not ok:
                    raise RuntimeError('Video insuficiente para la medición')
            alto, ancho = imagen.shape[:2]
            return cv2.resize(imagen, (960, int(alto * 960 / ancho)))
        modelo.track(siguiente())
        barrera.wait(timeout=90)
        proceso = psutil.Process()
        cpu_antes = proceso.cpu_times()
        inicio = time.perf_counter()
        for _ in range(frames):
            modelo.track(siguiente())
        duracion = time.perf_counter() - inicio
        cpu_despues = proceso.cpu_times()
        salida.put({
            'camara': indice, 'frames': frames, 'segundos': round(duracion, 3),
            'fps': round(frames / duracion, 3),
            'cpu_segundos': round(cpu_despues.user + cpu_despues.system - cpu_antes.user - cpu_antes.system, 3),
            'memoria_mb': round(proceso.memory_info().rss / 1024 ** 2, 1),
        })
        captura.release()
    except Exception as error:
        salida.put({'camara': indice, 'error': str(error)})
        try:
            barrera.abort()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--pesos', default='models/yolo26n.pt')
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--hilos', type=int, default=2)
    parser.add_argument('--camaras', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--salida', default='medicion_capacidad.json')
    args = parser.parse_args()
    assert Path(args.video).is_file() and Path(args.pesos).is_file()
    assert args.frames > 0 and args.hilos > 0 and all(n > 0 for n in args.camaras)
    contexto = multiprocessing.get_context('spawn')
    resultados = {'procesadores_logicos': os.cpu_count(), 'hilos_por_camara': args.hilos,
                  'ancho': 960, 'muestreo_objetivo_fps': 5, 'ensayos': []}
    for cantidad in args.camaras:
        cola = contexto.Queue()
        barrera = contexto.Barrier(cantidad)
        procesos = [contexto.Process(target=medir_camara,
                     args=(i + 1, args.video, args.pesos, args.frames, args.hilos, barrera, cola))
                    for i in range(cantidad)]
        for proceso in procesos:
            proceso.start()
        try:
            filas = [cola.get(timeout=120) for _ in procesos]
        finally:
            for proceso in procesos:
                proceso.join(timeout=5)
                if proceso.is_alive():
                    proceso.terminate()
                    proceso.join(timeout=2)
        ensayo = {'camaras_simultaneas': cantidad, 'resultados': filas}
        resultados['ensayos'].append(ensayo)
        print(json.dumps(ensayo, ensure_ascii=False), flush=True)
    Path(args.salida).write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
