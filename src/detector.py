"""
detector.py
Carga el modelo YOLO y ejecuta inferencia sobre frames de video.
No conoce nada de tracking, zonas, ni base de datos: solo recibe
un frame y devuelve las detecciones crudas.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from ultralytics import YOLO
from loguru import logger


@dataclass
class Detection:
    """Representa una detección individual en un frame."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class PedestrianDetector:
    """Envuelve el modelo YOLO para detección de peatones (clase 'person')."""

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        classes: List[int] | None = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes or [0]  # 0 = person en COCO
        self.device = device

        weights = Path(weights_path)
        if not weights.exists():
            logger.warning(
                f"No se encontró el modelo en {weights_path}. "
                f"Ultralytics intentará descargarlo automáticamente."
            )

        logger.info(f"Cargando modelo YOLO desde {weights_path} (device={device})")
        self.model = YOLO(weights_path)
        self.model.to(device)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Ejecuta inferencia sobre un frame y devuelve la lista de detecciones
        filtradas por clase y umbral de confianza.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append(
                Detection(
                    x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3],
                    confidence=conf, class_id=cls_id,
                )
            )

        return detections

    def count(self, frame: np.ndarray) -> int:
        """Atajo: devuelve solo la cantidad de peatones detectados en el frame."""
        return len(self.detect(frame))