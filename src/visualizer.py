"""
visualizer.py
Dibuja las detecciones/tracks sobre el frame y muestra una ventana
de debug local. NO se usa en producción (el pipeline real solo
envía datos al backend), es exclusivamente para desarrollo/pruebas.
"""

import cv2
import numpy as np
from loguru import logger


class Visualizer:
    """
    Dibuja bounding boxes, IDs de tracking y el polígono de zona
    sobre un frame, y lo muestra en una ventana local.
    """

    def __init__(
        self,
        window_name: str = "VisIAqp - Debug",
        show_boxes: bool = True,
        show_zone: bool = True,
    ):
        self.window_name = window_name
        self.show_boxes = show_boxes
        self.show_zone = show_zone

    def draw_detections(self, frame: np.ndarray, tracked_detections: list) -> np.ndarray:
        """Dibuja un box verde + track_id por cada detección trackeada."""
        output = frame.copy()

        if not self.show_boxes:
            return output

        for det in tracked_detections:
            pt1 = (int(det.x1), int(det.y1))
            pt2 = (int(det.x2), int(det.y2))
            cv2.rectangle(output, pt1, pt2, (0, 255, 0), 2)

            label = f"ID {det.track_id} ({det.confidence:.2f})"
            cv2.putText(
                output, label, (pt1[0], pt1[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )

        return output

    def draw_zone(self, frame: np.ndarray, zone, frame_width: int, frame_height: int) -> np.ndarray:
        """Dibuja el polígono de la zona de interés en azul."""
        output = frame.copy()

        if not self.show_zone:
            return output

        polygon = zone.polygon_for_frame(frame_width, frame_height)
        cv2.polylines(output, [polygon], isClosed=True, color=(255, 150, 0), thickness=2)

        return output

    def draw_info(self, frame: np.ndarray, count_in_zone: int, total_unique: int) -> np.ndarray:
        """Muestra contador de personas en zona y total acumulado, arriba a la izquierda."""
        output = frame.copy()
        text = f"En zona: {count_in_zone} | Total unicos: {total_unique}"
        cv2.putText(
            output, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )
        return output

    def show(self, frame: np.ndarray) -> bool:
        """
        Muestra el frame en una ventana. Devuelve False si el usuario
        presionó 'q' (señal para detener el loop principal).
        """
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q")

    def close(self) -> None:
        """Cierra todas las ventanas de OpenCV."""
        cv2.destroyAllWindows()
        logger.info("Ventanas de visualización cerradas")