"""
zones.py
Define y valida zonas de interés (polígonos) dentro del frame.
Recibe un punto (centro de una detección) y determina si cae
dentro de la zona configurada para esa cámara.

Los polígonos en config.yaml usan coordenadas normalizadas (0-1),
independientes de la resolución del frame, para que funcionen igual
sin importar el tamaño de video de cada cámara.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import cv2
from loguru import logger


@dataclass
class Zone:
    """
    Representa una zona de interés poligonal.

    id_zona: identificador que coincide con la tabla `zona` en la BD.
    polygon_normalized: lista de puntos (x, y) en rango 0-1, tal como
        vienen en config.yaml (independientes de resolución).
    """
    id_zona: int
    nombre: str
    polygon_normalized: List[Tuple[float, float]]

    def polygon_for_frame(self, frame_width: int, frame_height: int) -> np.ndarray:
        """
        Convierte el polígono normalizado a coordenadas de píxeles
        reales, según el tamaño del frame actual.
        """
        points = [
            (x * frame_width, y * frame_height)
            for x, y in self.polygon_normalized
        ]
        return np.array(points, dtype=np.int32)

    def contains_point(
        self,
        point: Tuple[float, float],
        frame_width: int,
        frame_height: int,
    ) -> bool:
        """
        Determina si un punto (ej. el centro de una detección, en
        coordenadas de píxeles) cae dentro de esta zona.
        """
        polygon = self.polygon_for_frame(frame_width, frame_height)
        result = cv2.pointPolygonTest(polygon, point, measureDist=False)
        return result >= 0  # >=0 significa dentro o sobre el borde


class ZoneManager:
    """
    Administra las zonas configuradas para una cámara y resuelve,
    para un conjunto de detecciones, a qué zona pertenece cada una.
    """

    def __init__(self, zones: List[Zone]):
        if not zones:
            logger.warning("ZoneManager inicializado sin zonas configuradas")
        self.zones = zones

    @classmethod
    def from_config(cls, camera_config: dict) -> "ZoneManager":
        """
        Construye un ZoneManager a partir de una entrada de `cameras`
        en config.yaml. Cada cámara tiene UNA zona asociada (id_zona)
        con su polígono (zone_polygon).
        """
        zone = Zone(
            id_zona=camera_config["id_zona"],
            nombre=camera_config.get("nombre", f"Zona {camera_config['id_zona']}"),
            polygon_normalized=[tuple(p) for p in camera_config["zone_polygon"]],
        )
        return cls(zones=[zone])

    def zone_for_point(
        self,
        point: Tuple[float, float],
        frame_width: int,
        frame_height: int,
    ) -> Zone | None:
        """
        Devuelve la primera zona que contiene el punto dado, o None
        si el punto no cae en ninguna zona configurada.
        """
        for zone in self.zones:
            if zone.contains_point(point, frame_width, frame_height):
                return zone
        return None

    def filter_detections_in_zone(
        self,
        detections: list,
        frame_width: int,
        frame_height: int,
    ) -> list:
        """
        Filtra una lista de TrackedDetection (de tracker.py), dejando
        solo las que caen dentro de alguna zona configurada.
        Asume que cada detección tiene un atributo `.center`.
        """
        in_zone = []
        for det in detections:
            zone = self.zone_for_point(det.center, frame_width, frame_height)
            if zone is not None:
                in_zone.append(det)
        return in_zone