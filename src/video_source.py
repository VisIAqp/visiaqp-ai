"""
video_source.py
Abstrae la fuente de video: puede ser un archivo local, una webcam
(índice entero) o un stream RTSP. Entrega frames uno a uno para
que detector.py los procese, sin que el resto del pipeline sepa
de dónde vienen.

El muestreo se controla con target_fps: en vez de un frame_skip fijo
(que depende del FPS original del video y no es comparable entre
fuentes distintas), se calcula automáticamente cuántos frames saltar
según el FPS real de cada fuente, para analizar siempre la misma
cantidad de imágenes por segundo de CONTENIDO, sin importar si el
video es de 24, 30 o 60 FPS.
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
      - str: ruta a un archivo de video, o URL RTSP/HTTP
    """

    def __init__(
        self,
        source: Union[int, str],
        resize_width: int | None = None,
        target_fps: float | None = None,
    ):
        self.source = source
        self.resize_width = resize_width
        self.target_fps = target_fps

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente de video: {source}")

        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0

        # Calcular frame_skip a partir del FPS real de la fuente y el target deseado.
        # Si no se puede leer el FPS real (algunos streams en vivo no lo reportan),
        # se asume 1 (no saltar) como fallback seguro.
        if self.target_fps and self.source_fps > 0:
            self.frame_skip = max(1, round(self.source_fps / self.target_fps))
        else:
            self.frame_skip = 1

        logger.info(
            f"Fuente de video abierta: {source} "
            f"(fps_origen={self.source_fps:.1f}, target_fps={self.target_fps}, "
            f"frame_skip_calculado={self.frame_skip})"
        )

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
        Generador que entrega frames uno a uno, aplicando el frame_skip
        calculado automáticamente a partir de target_fps y el FPS real
        de la fuente.
        """
        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.info(f"Fin de la fuente de video o fallo de lectura: {self.source}")
                break

            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue

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