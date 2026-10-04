"""
notifier.py
Gestiona la comunicación con el backend: obtiene la lista de cámaras,
gestiona el ciclo de vida de un `procesamiento`, y envía los resultados
por ventana. No conoce nada de detección, tracking ni zonas.
"""

import time
from typing import Optional

import requests
from loguru import logger


class Notifier:
    """
    Consulta cámaras, crea y cierra procesamientos, y envía resultados
    por ventana al backend, con reintentos ante fallos de red.
    """

    def __init__(
        self,
        api_base_url: str,
        endpoints: dict,
        api_token: Optional[str] = None,
        timeout_seconds: int = 5,
        retry_attempts: int = 3,
        retry_backoff_seconds: int = 2,
    ):
        self.api_base_url = api_base_url.rstrip("/")
        self.endpoints = endpoints
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.retry_backoff_seconds = retry_backoff_seconds

        self.headers = {"Content-Type": "application/json"}
        if api_token:
            self.headers["Authorization"] = f"Bearer {api_token}"

    def _request_with_retry(self, method: str, endpoint: str, payload: dict | None = None) -> dict | list | None:
        """
        Intenta la request dada, reintentando con backoff si falla.
        Devuelve el JSON de la respuesta si tuvo éxito, o None si agotó
        los intentos.
        """
        url = f"{self.api_base_url}{endpoint}"

        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = requests.request(
                    method, url, json=payload, headers=self.headers, timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                logger.info(f"{method} {url} exitoso (intento {attempt})")
                return response.json() if response.content else {}

            except requests.exceptions.RequestException as e:
                logger.warning(f"Intento {attempt}/{self.retry_attempts} falló para {method} {url}: {e}")
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_backoff_seconds * attempt)

        logger.error(f"No se pudo completar {method} {url} tras {self.retry_attempts} intentos")
        return None

    def get_cameras(self) -> list[dict] | None:
        """
        Obtiene la lista de cámaras activas desde el backend.
        Devuelve una lista de diccionarios con la forma que espera
        process_camera() en main.py (id_camara, id_zona, nombre, source,
        zone_polygon, event_start), o None si la consulta falló.
        """
        data = self._request_with_retry("GET", self.endpoints["listar_camaras"])
        if data is None:
            return None

        cameras = []
        for cam in data:
            cameras.append({
                "id_camara": cam["id_camara"],
                "id_zona": cam["id_zona"],
                "nombre": cam.get("nombre", f"Camara {cam['id_camara']}"),
                "source": cam["url_stream"],
                "zone_polygon": cam.get("zone_polygon"),
                # event_start no aplica a cámaras en vivo; solo tendría sentido
                # si el backend algún día expone videos pregrabados por esta vía.
            })
        return cameras

    def create_procesamiento(self, payload: dict) -> int | None:
        """
        Crea un procesamiento al iniciar una cámara. Devuelve el
        id_procesamiento asignado por el backend, o None si falló.
        """
        data = self._request_with_retry("POST", self.endpoints["crear_procesamiento"], payload)
        if data is None:
            return None
        return data.get("id_procesamiento")

    def close_procesamiento(self, id_procesamiento: int, payload: dict) -> bool:
        """Cierra un procesamiento (estado completado/error, fecha_fin_proceso)."""
        endpoint = self.endpoints["cerrar_procesamiento"].format(id=id_procesamiento)
        return self._request_with_retry("PATCH", endpoint, payload) is not None

    def send_resultado(self, id_procesamiento: int, resultado) -> bool:
        """Envía el ResultadoDeteccion de una ventana ya cerrada."""
        endpoint = self.endpoints["resultado"].format(id=id_procesamiento)
        return self._request_with_retry("POST", endpoint, resultado.to_dict()) is not None