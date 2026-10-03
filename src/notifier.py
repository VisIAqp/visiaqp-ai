"""
notifier.py
Envía los resultados calculados por analyzer.py al backend FastAPI.
No conoce nada de detección, tracking ni zonas: solo recibe objetos
ya armados (ResultadoDeteccion, Indicador) y los manda por HTTP.
"""

import time
from typing import Optional

import requests
from loguru import logger


class Notifier:
    """
    Envía resultados e indicadores al backend, con reintentos ante
    fallos de red. Si todos los reintentos fallan, devuelve False y
    quien lo llame (main.py) decide si guarda el dato localmente.
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
        self.endpoints = endpoints  # {"resultado": "/api/...", "indicador": "/api/..."}
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.retry_backoff_seconds = retry_backoff_seconds

        self.headers = {"Content-Type": "application/json"}
        if api_token:
            self.headers["Authorization"] = f"Bearer {api_token}"

    def _post_with_retry(self, endpoint: str, payload: dict) -> bool:
        """
        Intenta hacer POST al endpoint dado, reintentando con backoff
        si falla. Devuelve True si tuvo éxito, False si agotó los intentos.
        """
        url = f"{self.api_base_url}{endpoint}"

        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = requests.post(
                    url, json=payload, headers=self.headers, timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                logger.info(f"Notificación enviada a {url} (intento {attempt})")
                return True

            except requests.exceptions.RequestException as e:
                logger.warning(f"Intento {attempt}/{self.retry_attempts} falló para {url}: {e}")
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_backoff_seconds * attempt)  # backoff incremental

        logger.error(f"No se pudo enviar la notificación a {url} tras {self.retry_attempts} intentos")
        return False

    def send_resultado(self, resultado) -> bool:
        """Envía un ResultadoDeteccion al endpoint configurado en endpoints['resultado']."""
        return self._post_with_retry(self.endpoints["resultado"], resultado.to_dict())

    def send_indicador(self, indicador) -> bool:
        """Envía un Indicador al endpoint configurado en endpoints['indicador']."""
        return self._post_with_retry(self.endpoints["indicador"], indicador.to_dict())