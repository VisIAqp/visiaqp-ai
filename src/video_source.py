"""
video_source.py
Abstrae la fuente de video: puede ser un archivo local, una webcam
(índice entero) o un stream RTSP/HTTP. Entrega frames uno a uno para
que el pipeline los procese, sin que el resto sepa de dónde vienen.

El muestreo se controla con target_fps: en vez de un frame_skip fijo
(que depende del FPS original y no es comparable entre fuentes), se
calcula cuántos frames saltar según el FPS real de cada fuente, para
analizar siempre la misma cantidad de imágenes por segundo de CONTENIDO.

Cada frame se entrega junto con los segundos transcurridos en la fuente:
  - archivo: posición real dentro del video (frame / fps), independiente
    de qué tan rápido o lento procese la CPU.
  - stream en vivo: segundos de reloj desde que se abrió la fuente.
"""

import time
from typing import Iterator, Tuple, Union

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

    # Todo lo que empiece así se trata como fuente en vivo (no como archivo)
    LIVE_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")

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

        self.is_live = isinstance(source, int) or str(source).lower().startswith(self.LIVE_PREFIXES)
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0

        # Calcular frame_skip a partir del FPS real de la fuente y el target deseado.
        # Si no se puede leer el FPS real (algunos streams en vivo no lo reportan),
        # se asume 1 (no saltar) como fallback seguro.
        if self.target_fps and self.source_fps > 0:
            self.frame_skip = max(1, round(self.source_fps / self.target_fps))
        else:
            self.frame_skip = 1

        if not self.is_live and self.source_fps <= 0:
            logger.warning(
                f"No se pudo leer el FPS del archivo {source}; "
                f"el tiempo del evento se estimará con el reloj del sistema."
            )

        logger.info(
            f"Fuente de video abierta: {source} "
            f"(en_vivo={self.is_live}, fps_origen={self.source_fps:.1f}, "
            f"target_fps={self.target_fps}, frame_skip_calculado={self.frame_skip})"
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

    def _elapsed_seconds(self, frame_count: int, start_monotonic: float) -> float:
        """
        Segundos transcurridos en la fuente al leer este frame.
        Archivo con FPS conocido: posición real en el video.
        En vivo (o FPS desconocido): tiempo de reloj desde la apertura.
        """
        if not self.is_live and self.source_fps > 0:
            return frame_count / self.source_fps
        return time.monotonic() - start_monotonic

    def frames(self) -> Iterator[Tuple[np.ndarray, float]]:
        """
        Generador que entrega tuplas (frame, segundos_transcurridos),
        aplicando el frame_skip calculado a partir de target_fps.
        """
        frame_count = 0
        start_monotonic = time.monotonic()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.info(f"Fin de la fuente de video o fallo de lectura: {self.source}")
                break

            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue

            elapsed = self._elapsed_seconds(frame_count, start_monotonic)
            yield self._resize(frame), elapsed

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