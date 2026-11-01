"""
analyzer.py
Agrega las detecciones trackeadas de una zona en ventanas de tiempo y
produce un ResultadoDeteccion crudo por cada ventana cerrada. No conoce
nada de HTTP ni de clasificación bajo/medio/alto: esa clasificación la
hace el backend (con umbral_concentracion, por zona) a partir de estos
resultados crudos, no el motor.

Todos los timestamps son de EVENTO (cuándo ocurrió lo que muestra el
video) y los provee quien orquesta; el analyzer nunca consulta el reloj
del sistema, para que los resultados no dependan de la velocidad de
procesamiento.
"""

from dataclasses import dataclass, field
from uuid import uuid4
from datetime import datetime, timedelta
from typing import List

from loguru import logger


@dataclass
class ResultadoDeteccion:
    """Resultado agregado de UNA ventana de análisis (no de un frame)."""
    id_zona: int
    id_camara: int | None
    periodo_inicio: str   # ISO datetime (evento)
    periodo_fin: str      # ISO datetime (evento)
    cantidad_peatones: int       # IDs distintos observados dentro de la zona en la ventana
    confianza_promedio: float

    promedio_presentes: float | None = None
    max_presentes: int | None = None
    segundos_observados: float | None = None
    clave_ventana: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict:
        return {
            "clave_ventana": self.clave_ventana,
            "id_zona": self.id_zona,
            "id_camara": self.id_camara,
            "periodo_inicio": self.periodo_inicio,
            "periodo_fin": self.periodo_fin,
            "cantidad_peatones": self.cantidad_peatones,
            "confianza_promedio": round(self.confianza_promedio, 2),
            "promedio_presentes": self.promedio_presentes,
            "max_presentes": self.max_presentes,
            "segundos_observados": self.segundos_observados,
        }


@dataclass
class _WindowSample:
    """Muestra interna: una lectura de conteo en un instante, dentro de la ventana actual."""
    timestamp: datetime
    count_in_zone: int
    avg_confidence: float


class ZoneAnalyzer:
    """
    Acumula muestras de conteo por zona a lo largo de una ventana de
    tiempo (window_seconds) y produce un ResultadoDeteccion al cerrarla.
    """

    def __init__(self, id_zona: int, window_seconds: int = 60):
        self.id_zona = id_zona
        self.window_seconds = window_seconds

        self._samples: List[_WindowSample] = []
        self._ids_en_zona: set[int] = set()
        self._window_start: datetime | None = None

    def add_sample(self, tracked_detections_in_zone: list, timestamp: datetime) -> None:
        """
        Registra una muestra: la cantidad de personas detectadas EN ZONA
        en el instante `timestamp` (hora del evento), junto con su
        confianza promedio.
        """
        if self._samples and timestamp < self._samples[-1].timestamp:
            raise ValueError("Las muestras deben llegar en orden temporal")
        if self._window_start is None:
            self._window_start = timestamp

        self._ids_en_zona.update(d.track_id for d in tracked_detections_in_zone)
        count = len(tracked_detections_in_zone)
        avg_conf = (
            sum(d.confidence for d in tracked_detections_in_zone) / count
            if count > 0 else 0.0
        )

        self._samples.append(_WindowSample(timestamp=timestamp, count_in_zone=count, avg_confidence=avg_conf))

    def has_pending_samples(self) -> bool:
        """Indica si hay muestras acumuladas en la ventana actual sin cerrar."""
        return len(self._samples) > 0

    def window_elapsed(self, now: datetime) -> bool:
        """
        Indica si ya pasaron window_seconds de tiempo de EVENTO desde que
        empezó la ventana actual. `now` es el timestamp de evento actual.
        """
        if self._window_start is None:
            return False
        return (now - self._window_start) >= timedelta(seconds=self.window_seconds)

    def close_window(self, id_camara: int | None) -> ResultadoDeteccion | None:
        """
        Cierra la ventana actual, construye el ResultadoDeteccion crudo
        de esa ventana, y reinicia el estado para la siguiente.

        Cuenta únicamente los IDs recibidos tras el filtro espacial.
        No representa visitantes únicos entre ventanas ni entre cámaras.
        """
        if not self._samples or self._window_start is None:
            logger.warning(f"close_window() llamado sin muestras para zona {self.id_zona}")
            return None

        periodo_inicio = self._window_start
        periodo_fin = self._samples[-1].timestamp
        avg_conf = sum(s.avg_confidence for s in self._samples) / len(self._samples)

        # Aproximación por escalones: cada lectura rige hasta la siguiente.
        # Se usa tiempo del evento, no velocidad de inferencia ni número de frames.
        segundos = (periodo_fin - periodo_inicio).total_seconds()
        personas_segundo = sum(
            anterior.count_in_zone * (actual.timestamp - anterior.timestamp).total_seconds()
            for anterior, actual in zip(self._samples, self._samples[1:])
        )
        resultado = ResultadoDeteccion(
            id_zona=self.id_zona,
            id_camara=id_camara,
            periodo_inicio=periodo_inicio.isoformat(),
            periodo_fin=periodo_fin.isoformat(),
            cantidad_peatones=len(self._ids_en_zona),
            confianza_promedio=avg_conf,
            promedio_presentes=personas_segundo / segundos if segundos > 0 else None,
            max_presentes=max(s.count_in_zone for s in self._samples) if segundos > 0 else None,
            segundos_observados=segundos if segundos > 0 else None,
        )

        logger.info(
            f"Ventana cerrada [zona={self.id_zona}]: "
            f"cantidad_peatones={resultado.cantidad_peatones}, "
            f"confianza_promedio={resultado.confianza_promedio:.2f}"
        )

        # reset para la siguiente ventana
        self._samples = []
        self._ids_en_zona.clear()
        self._window_start = None

        return resultado