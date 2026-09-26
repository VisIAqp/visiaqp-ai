"""
tracker.py
Aplica tracking (ByteTrack) sobre las detecciones para evitar doble
conteo entre frames. Usa el tracking nativo de Ultralytics, que ya
incluye ByteTrack sin dependencias adicionales.

A diferencia de detector.py (que solo detecta frame a frame sin memoria),
este módulo mantiene identidad de cada persona a través del tiempo,
asignándole un track_id estable mientras siga visible (o dentro del
buffer configurado).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

import numpy as np
from ultralytics import YOLO
from loguru import logger


@dataclass
class TrackedDetection:
    """Una detección con identidad persistente entre frames."""
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class PedestrianTracker:
    """
    Envuelve el tracking nativo de Ultralytics (ByteTrack) sobre el
    mismo modelo YOLO usado para detección. Mantiene un registro de
    IDs únicos vistos en la sesión para conteo acumulado real.
    """

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        classes: List[int] | None = None,
        algorithm: str = "bytetrack",
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes or [0]  # 0 = person en COCO
        self.device = device

        # Ultralytics trae dos configs built-in: "bytetrack.yaml" y "botsort.yaml"
        tracker_map = {
            "bytetrack": "bytetrack.yaml",
            "deepsort": "botsort.yaml",  # fallback razonable si se pide deepsort sin instalarlo
        }
        self.tracker_config = tracker_map.get(algorithm, "bytetrack.yaml")
        if algorithm == "deepsort":
            logger.warning(
                "DeepSORT no está instalado como dependencia extra; "
                "usando BoT-SORT (incluido en Ultralytics) como alternativa."
            )

        # Nota: track_thresh, track_buffer y match_thresh son parámetros de
        # bytetrack.yaml. Ultralytics no permite pasarlos directo por código
        # sin generar un YAML custom; se documentan aquí para referencia y
        # se pueden ajustar creando una copia de bytetrack.yaml si hace falta
        # afinar el comportamiento más adelante.
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh

        weights = Path(weights_path)
        if not weights.exists():
            logger.warning(
                f"No se encontró el modelo en {weights_path}. "
                f"Ultralytics intentará descargarlo automáticamente."
            )

        logger.info(f"Cargando modelo YOLO para tracking desde {weights_path} (device={device})")
        self.model = YOLO(weights_path)
        self.model.to(device)

        # IDs únicos vistos desde que se creó este tracker (conteo acumulado real)
        self._seen_track_ids: Set[int] = set()

    def track(self, frame: np.ndarray) -> List[TrackedDetection]:
        """
        Ejecuta detección + tracking sobre un frame y devuelve la lista
        de detecciones con track_id persistente.

        IMPORTANTE: persist=True es necesario para que el tracker recuerde
        el estado entre llamadas sucesivas de este metodo (frame a frame).
        Si se llama con persist=False el tracking se resetea cada vez.
        """
        results = self.model.track(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            tracker=self.tracker_config,
            persist=True,
            verbose=False,
        )

        tracked: List[TrackedDetection] = []
        if not results:
            return tracked

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            # boxes.id es None cuando el tracker aún no asignó IDs
            # (puede pasar en el primer frame o si no hay detecciones)
            return tracked

        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            track_id = int(box.id[0])

            self._seen_track_ids.add(track_id)

            tracked.append(
                TrackedDetection(
                    track_id=track_id,
                    x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3],
                    confidence=conf, class_id=cls_id,
                )
            )

        return tracked

    def current_count(self, frame: np.ndarray) -> int:
        """Cantidad de personas visibles EN ESTE frame (no acumulado)."""
        return len(self.track(frame))

    def total_unique_count(self) -> int:
        """
        Cantidad TOTAL de personas únicas vistas desde que se creó este
        tracker (acumulado). Útil para saber cuántas personas distintas
        pasaron por la zona, no solo cuántas hay en un instante.
        """
        return len(self._seen_track_ids)

    def reset(self) -> None:
        """Reinicia el conteo acumulado y el estado del tracker."""
        self._seen_track_ids.clear()
        # Forzar reset del tracker interno de Ultralytics
        if hasattr(self.model.predictor, "trackers"):
            self.model.predictor.trackers = None
        logger.info("Tracker reiniciado")