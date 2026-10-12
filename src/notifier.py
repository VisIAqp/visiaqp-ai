"""
notifier.py
Comunicación con el backend: cámaras, ciclo de vida de un `procesamiento`,
resultados por ventana y catálogo de videos pregrabados. No conoce nada de
detección, tracking ni zonas.

Reintentos: se reintentan errores de red, 5xx, 408 y 429. El resto de los
4xx (400, 404, 409, 422...) se devuelven de inmediato: repetirlos no cambia
el resultado, y el 409 de "reclamar" significa que otro motor ya lo tomó.
"""

import time
from dataclasses import dataclass
from typing import Optional

import requests
from loguru import logger

RETRYABLE_4XX = {408, 429}


@dataclass
class _ApiResult:
    ok: bool
    status: int | None = None
    data: dict | list | None = None


class Notifier:
    def __init__(
        self,
        api_base_url: str,
        endpoints: dict,
        api_key: Optional[str] = None,
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
        if api_key:
            self.headers["X-API-Key"] = api_key

    def _request(self, method: str, endpoint: str, payload: dict | None = None) -> _ApiResult:
        """Ejecuta la request con reintentos y devuelve el resultado completo (incluye el status)."""
        url = f"{self.api_base_url}{endpoint}"
        last_status = None

        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = requests.request(
                    method, url, json=payload, headers=self.headers, timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                logger.info(f"{method} {url} exitoso (intento {attempt})")
                data = response.json() if response.content else {}
                return _ApiResult(ok=True, status=response.status_code, data=data)

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                last_status = status
                if status is not None and 400 <= status < 500 and status not in RETRYABLE_4XX:
                    detalle = (e.response.text or "")[:200] if e.response is not None else ""
                    logger.warning(f"{method} {url} rechazado con {status}; no se reintenta. {detalle}")
                    return _ApiResult(ok=False, status=status)
                logger.warning(f"Intento {attempt}/{self.retry_attempts} falló para {method} {url}: {e}")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Intento {attempt}/{self.retry_attempts} falló para {method} {url}: {e}")

            if attempt < self.retry_attempts:
                time.sleep(self.retry_backoff_seconds * attempt)

        logger.error(f"No se pudo completar {method} {url} tras {self.retry_attempts} intentos")
        return _ApiResult(ok=False, status=last_status)

    def _request_with_retry(self, method: str, endpoint: str, payload: dict | None = None) -> dict | list | None:
        """Atajo: devuelve el JSON de la respuesta si tuvo éxito, o None."""
        result = self._request(method, endpoint, payload)
        return result.data if result.ok else None

    # ---------------- Cámaras en vivo ----------------

    def get_cameras(self) -> list[dict] | None:
        """
        Lista de cámaras activas desde el backend, con la forma que espera
        process_camera() (id_camara, id_zona, nombre, source, zone_polygon),
        o None si la consulta falló.
        """
        data = self._request_with_retry("GET", self.endpoints["listar_camaras"])
        if data is None:
            return None

        return [
            {
                "id_camara": cam["id_camara"],
                "id_zona": cam["id_zona"],
                "nombre": cam.get("nombre", f"Camara {cam['id_camara']}"),
                "source": cam["url_stream"],
                "zone_polygon": cam.get("zone_polygon"),
            }
            for cam in data
        ]

    # ---------------- Ciclo de vida del procesamiento ----------------

    def create_procesamiento(self, payload: dict) -> int | None:
        """Crea un procesamiento. Devuelve su id_procesamiento, o None si falló."""
        data = self._request_with_retry("POST", self.endpoints["crear_procesamiento"], payload)
        if data is None:
            return None
        return data.get("id_procesamiento")

    def close_procesamiento(self, id_procesamiento: int, payload: dict) -> bool:
        """Cierra un procesamiento (estado completado/error, fecha_fin_proceso)."""
        endpoint = self.endpoints["cerrar_procesamiento"].format(id=id_procesamiento)
        return self._request("PATCH", endpoint, payload).ok

    def send_resultado(self, id_procesamiento: int, resultado) -> bool:
        """Envía el ResultadoDeteccion de una ventana ya cerrada."""
        endpoint = self.endpoints["resultado"].format(id=id_procesamiento)
        return self._request("POST", endpoint, resultado.to_dict()).ok

    # ---------------- Videos pregrabados ----------------

    def sync_videos(self, videos: list[dict]) -> bool:
        """Sincroniza el catálogo: envía la lista COMPLETA de archivos que existen hoy."""
        result = self._request("POST", self.endpoints["sincronizar_videos"], {"videos": videos})
        if not result.ok:
            return False
        if isinstance(result.data, dict):
            logger.info(
                f"Catálogo sincronizado: registrados={result.data.get('registrados')}, "
                f"actualizados={result.data.get('actualizados')}, "
                f"no_disponibles={result.data.get('no_disponibles')}"
            )
        return True

    def get_pending(self) -> list[dict] | None:
        """Procesamientos pendientes (el más antiguo primero), o None si la consulta falló."""
        result = self._request("GET", self.endpoints["listar_pendientes"])
        if not result.ok:
            return None
        if not isinstance(result.data, list):
            logger.error("La respuesta de pendientes no es una lista; se ignora")
            return None
        return result.data

    def claim_procesamiento(self, id_procesamiento: int) -> str:
        """
        Reclama un procesamiento pendiente (pasa a 'procesando').
        Devuelve "ok", "conflict" (409: ya lo tomó otra instancia) o "error".
        El 409 nunca se reintenta.
        """
        endpoint = self.endpoints["iniciar_procesamiento"].format(id=id_procesamiento)
        result = self._request("PATCH", endpoint)
        if result.ok:
            return "ok"
        if result.status == 409:
            return "conflict"
        return "error"