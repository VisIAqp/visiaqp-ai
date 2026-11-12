"""Fuentes de archivo y captura en vivo con un único frame pendiente."""
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Iterator

import cv2
import numpy as np
from loguru import logger


def ahora_lima() -> datetime:
    return datetime.now(timezone(timedelta(hours=-5))).replace(tzinfo=None)


class ErrorTransmision(RuntimeError):
    """Lectura interrumpida o cámara sin imágenes recientes."""


class CapturaBloqueada(ErrorTransmision):
    """El proceso debe terminar para liberar una captura que no responde."""


class VideoSource:
    LIVE_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")

    @classmethod
    def es_fuente_en_vivo(cls, source):
        return isinstance(source, int) or str(source).lower().startswith(cls.LIVE_PREFIXES)

    def __init__(self, source, resize_width=None, target_fps=None,
                 read_timeout_seconds=10, stop_event=None):
        self.source = source
        self.resize_width = resize_width
        self.target_fps = target_fps
        self.read_timeout_seconds = read_timeout_seconds
        self.stop_event = stop_event or threading.Event()
        self.is_live = self.es_fuente_en_vivo(source)
        self._parar = threading.Event()
        self._condicion = threading.Condition()
        self._ultimo = None
        self._secuencia = 0
        self._error = False
        self._lector = None
        self._bloqueada = False
        # Hora de recepción local. No se conoce la hora del sensor remoto.
        self.started_at = ahora_lima()
        self.start_monotonic = time.monotonic()
        self.last_captured_monotonic = self.start_monotonic
        if self.is_live and not isinstance(source, int):
            tiempo_ms = int(read_timeout_seconds * 1000)
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, tiempo_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, tiempo_ms,
            ])
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap.release()
            raise ErrorTransmision("No se pudo abrir la fuente de video")
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.frame_skip = max(1, round(self.source_fps / target_fps)) if target_fps and self.source_fps > 0 else 1
        if not self.is_live and self.source_fps <= 0:
            self.cap.release()
            raise ValueError("El archivo no informa FPS: no se puede determinar su tiempo de evento")
        if self.is_live:
            self._lector = threading.Thread(target=self._capturar, daemon=True)
            self._lector.start()

    def _capturar(self):
        try:
            while not self._parar.is_set() and not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    break
                instante = time.monotonic()
                with self._condicion:
                    self._ultimo = (frame, instante)
                    self._secuencia += 1
                    self._condicion.notify_all()
        finally:
            self.cap.release()
            with self._condicion:
                self._error = True
                self._condicion.notify_all()

    def _resize(self, frame):
        if not self.resize_width or frame.shape[1] == self.resize_width:
            return frame
        h, w = frame.shape[:2]
        return cv2.resize(frame, (self.resize_width, int(h * self.resize_width / w)), interpolation=cv2.INTER_LINEAR)

    def _elapsed_seconds(self, frame_count, start_monotonic):
        if not self.is_live and self.source_fps > 0:
            return frame_count / self.source_fps
        return time.monotonic() - start_monotonic

    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        try:
            if not self.is_live:
                contador = 0
                while not self.stop_event.is_set():
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    contador += 1
                    if contador % self.frame_skip == 0:
                        yield self._resize(frame), contador / self.source_fps
                return
            secuencia = 0
            siguiente = 0.0
            while not self.stop_event.is_set() and not self._parar.is_set():
                with self._condicion:
                    self._condicion.wait_for(
                        lambda: self._secuencia != secuencia or self._error or self.stop_event.is_set(),
                        timeout=0.5,
                    )
                    if self._error:
                        raise ErrorTransmision("La transmisión dejó de entregar imágenes")
                    if time.monotonic() - (self._ultimo[1] if self._ultimo else self.start_monotonic) > self.read_timeout_seconds:
                        raise ErrorTransmision("La cámara no entrega imágenes recientes")
                    if self._secuencia == secuencia:
                        continue
                    espera = siguiente - time.monotonic()
                    if espera <= 0:
                        frame, instante = self._ultimo
                        secuencia = self._secuencia
                        self.last_captured_monotonic = instante
                if espera > 0:
                    self.stop_event.wait(min(espera, 0.5))
                    continue
                siguiente = time.monotonic() + (1 / self.target_fps if self.target_fps else 0)
                yield self._resize(frame), instante - self.start_monotonic
        finally:
            self.release()

    def release(self):
        if self._bloqueada:
            raise CapturaBloqueada("La captura no responde al cierre")
        self._parar.set()
        if self._lector:
            self._lector.join(timeout=self.read_timeout_seconds + 1)
            if self._lector.is_alive():
                self._bloqueada = True
                raise CapturaBloqueada("La captura no responde al cierre")
        elif self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
