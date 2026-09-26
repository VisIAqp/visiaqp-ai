"""
video_source.py
Abstrae la fuente de video: puede ser un archivo local, una webcam
(índice entero) o un stream RTSP. Entrega frames uno a uno para
que detector.py los procese, sin que el resto del pipeline sepa
de dónde vienen.
"""

import time
from typing import Iterator, Union

import cv2
import numpy as np
from loguru import logger


class VideoSource:
    """
    Envuelve cv2.VideoCapture para leer frames de una fuente de video.

    source puede ser:
      - int: índice de webcam (ej. 0)
      - str: ruta a un archivo de video, o URL RTSP
    """

    def __init__(
        self,
        source: Union[int, str],
        resize_width: int | None = None,
        frame_skip: int = 1,
        max_fps: int | None = None,
    ):
        self.source = source
        self.resize_width = resize_width
        self.frame_skip = max(1, frame_skip)
        self.max_fps = max_fps
        self._min_frame_interval = (1.0 / max_fps) if max_fps else 0.0

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente de video: {source}")

        logger.info(f"Fuente de video abierta: {source}")

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """Redimensiona el frame manteniendo el aspect ratio, si resize_width está definido."""
        if not self.resize_width:
            return frame

        h, w = frame.shape[:2]
        if w == self.resize_width:
            return frame

        ratio = self.resize_width / float(w)
        new_size = (self.resize_width, int(h * ratio))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)

    def frames(self) -> Iterator[np.ndarray]:
        """
        Generador que entrega frames uno a uno, aplicando frame_skip
        (para no procesar cada frame, mejora rendimiento) y limitando
        la tasa según max_fps si está configurado.
        """
        frame_count = 0
        last_yield_time = 0.0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.info(f"Fin de la fuente de video o fallo de lectura: {self.source}")
                break

            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue

            if self._min_frame_interval:
                elapsed = time.time() - last_yield_time
                if elapsed < self._min_frame_interval:
                    time.sleep(self._min_frame_interval - elapsed)
                last_yield_time = time.time()

            yield self._resize(frame)

        self.release()

    def release(self) -> None:
        """Libera el recurso de video. Seguro llamar más de una vez."""
        if self.cap.isOpened():
            self.cap.release()
            logger.info(f"Fuente de video liberada: {self.source}")

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()