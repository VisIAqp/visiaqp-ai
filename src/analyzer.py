"""
analyzer.py
Calcula densidad/concentración de peatones por zona, agregando las
detecciones trackeadas a lo largo de una ventana de tiempo. No conoce
nada de HTTP ni de la API: solo produce los objetos de datos listos
para que notifier.py los envíe.

Todos los timestamps son de EVENTO (cuándo ocurrió lo que muestra el
video) y los provee quien orquesta; el analyzer nunca consulta el reloj
del sistema, para que los resultados no dependan de la velocidad de
procesamiento.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

from loguru import logger


@dataclass
class ResultadoDeteccion:
    """Resultado de un procesamiento puntual (una ventana de análisis)."""
    id_zona: int
    id_camara: int | None
    tipo_fuente: str  # "imagen" | "video"
    fecha_analisis: str   # "YYYY-MM-DD" (fecha del EVENTO)
    hora_analisis: str    # "HH:MM:SS"   (hora del EVENTO)
    cantidad_peatones: int
    confianza_promedio: float

    def to_dict(self) -> dict:
        return {
            "id_zona": self.id_zona,
            "id_camara": self.id_camara,
            "tipo_fuente": self.tipo_fuente,
            "fecha_analisis": self.fecha_analisis,
            "hora_analisis": self.hora_analisis,
            "cantidad_peatones": self.cantidad_peatones,
            "confianza_promedio": round(self.confianza_promedio, 2),
        }


@dataclass
class Indicador:
    """Indicador agregado de concentración para una ventana de tiempo."""
    id_zona: int
    periodo_inicio: str  # ISO datetime (evento)
    periodo_fin: str      # ISO datetime (evento)
    total_peatones: int
    promedio_peatones: float
    nivel_concentracion: str  # "bajo" | "medio" | "alto"

    def to_dict(self) -> dict:
        return {
            "id_zona": self.id_zona,
            "periodo_inicio": self.periodo_inicio,
            "periodo_fin": self.periodo_fin,
            "total_peatones": self.total_peatones,
            "promedio_peatones": round(self.promedio_peatones, 2),
            "nivel_concentracion": self.nivel_concentracion,
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
    tiempo (window_seconds) y calcula el indicador de concentración
    al cerrar cada ventana.
    """

    def __init__(
        self,
        id_zona: int,
        window_seconds: int = 60,
        thresholds: dict | None = None,
    ):
        self.id_zona = id_zona
        self.window_seconds = window_seconds
        # thresholds: {"bajo": 10, "medio": 30} -> < bajo => "bajo", < medio => "medio", resto "alto"
        self.thresholds = thresholds or {"bajo": 10, "medio": 30}

        self._samples: List[_WindowSample] = []
        self._window_start: datetime | None = None

    def add_sample(self, tracked_detections_in_zone: list, timestamp: datetime) -> None:
        """
        Registra una muestra: la cantidad de personas detectadas EN ZONA
        en el instante `timestamp` (hora del evento), junto con su
        confianza promedio.
        """
        if self._window_start is None:
            self._window_start = timestamp

        count = len(tracked_detections_in_zone)
        avg_conf = (
            sum(d.confidence for d in tracked_detections_in_zone) / count
            if count > 0 else 0.0
        )

        self._samples.append(_WindowSample(timestamp=timestamp, count_in_zone=count, avg_confidence=avg_conf))

    def has_pending_samples(self) -> bool:
        """Indica si hay muestras acumuladas en la ventana actual sin cerrar."""
        return len(self._samples) > 0

    def build_resultado(
        self,
        id_camara: int | None,
        tipo_fuente: str,
        cantidad_peatones_unicos: int,
        timestamp: datetime,
    ) -> ResultadoDeteccion:
        """
        Construye un ResultadoDeteccion (para la tabla resultado_deteccion)
        con la fecha/hora de EVENTO `timestamp` (normalmente el fin de la ventana).

        IMPORTANTE: llamar ANTES de close_window(), que limpia las muestras
        usadas aquí para calcular la confianza promedio.
        """
        avg_conf = (
            sum(s.avg_confidence for s in self._samples) / len(self._samples)
            if self._samples else 0.0
        )

        return ResultadoDeteccion(
            id_zona=self.id_zona,
            id_camara=id_camara,
            tipo_fuente=tipo_fuente,
            fecha_analisis=timestamp.strftime("%Y-%m-%d"),
            hora_analisis=timestamp.strftime("%H:%M:%S"),
            cantidad_peatones=cantidad_peatones_unicos,
            confianza_promedio=avg_conf,
        )

    def window_elapsed(self, now: datetime) -> bool:
        """
        Indica si ya pasaron window_seconds de tiempo de EVENTO desde que
        empezó la ventana actual. `now` es el timestamp de evento actual.
        """
        if self._window_start is None:
            return False
        return (now - self._window_start) >= timedelta(seconds=self.window_seconds)

    def _classify(self, promedio_peatones: float) -> str:
        """Clasifica el promedio de peatones según los thresholds configurados."""
        if promedio_peatones < self.thresholds["bajo"]:
            return "bajo"
        elif promedio_peatones < self.thresholds["medio"]:
            return "medio"
        return "alto"

    def close_window(self, cantidad_peatones_unicos_totales: int) -> Indicador | None:
        """
        Cierra la ventana actual, calcula el indicador de concentración,
        y reinicia el estado para la siguiente ventana.

        cantidad_peatones_unicos_totales: viene de tracker.window_unique_count(),
        representa personas ÚNICAS distintas vistas en la ventana (no promedio
        de conteos instantáneos, que sobrestimaría si la misma persona se
        queda varios frames).
        """
        if not self._samples or self._window_start is None:
            logger.warning(f"close_window() llamado sin muestras para zona {self.id_zona}")
            return None

        periodo_inicio = self._window_start
        periodo_fin = self._samples[-1].timestamp

        promedio = sum(s.count_in_zone for s in self._samples) / len(self._samples)
        nivel = self._classify(promedio)

        indicador = Indicador(
            id_zona=self.id_zona,
            periodo_inicio=periodo_inicio.isoformat(),
            periodo_fin=periodo_fin.isoformat(),
            total_peatones=cantidad_peatones_unicos_totales,
            promedio_peatones=promedio,
            nivel_concentracion=nivel,
        )

        logger.info(
            f"Ventana cerrada [zona={self.id_zona}]: "
            f"total={indicador.total_peatones}, promedio={indicador.promedio_peatones:.2f}, "
            f"nivel={indicador.nivel_concentracion}"
        )

        # reset para la siguiente ventana
        self._samples = []
        self._window_start = None

        return indicador