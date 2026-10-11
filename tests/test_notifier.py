import pytest
import requests

from src.notifier import Notifier

ENDPOINTS = {
    "listar_camaras": "/api/camaras",
    "crear_procesamiento": "/api/procesamientos",
    "cerrar_procesamiento": "/api/procesamientos/{id}",
    "resultado": "/api/procesamientos/{id}/resultados",
    "sincronizar_videos": "/api/videos/sincronizar",
    "listar_pendientes": "/api/procesamientos/pendientes",
    "iniciar_procesamiento": "/api/procesamientos/{id}/iniciar",
}


class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = {} if json_data is None else json_data
        self.content = b"{}"
        self.text = ""

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)


def patch_requests(monkeypatch, respuestas):
    """Reemplaza requests.request. Si el notifier hace más llamadas que respuestas, el test falla."""
    llamadas = []
    it = iter(respuestas)

    def fake_request(method, url, **kwargs):
        llamadas.append((method, url, kwargs))
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr("src.notifier.requests.request", fake_request)
    return llamadas


@pytest.fixture
def notifier():
    return Notifier("http://api", ENDPOINTS, api_key="clave", timeout_seconds=1,
                    retry_attempts=3, retry_backoff_seconds=0)


def test_envia_header_x_api_key(notifier, monkeypatch):
    llamadas = patch_requests(monkeypatch, [FakeResponse(200)])
    notifier.claim_procesamiento(5)
    assert llamadas[0][2]["headers"]["X-API-Key"] == "clave"


def test_claim_ok(notifier, monkeypatch):
    llamadas = patch_requests(monkeypatch, [FakeResponse(200, {"id_procesamiento": 5, "estado": "procesando"})])
    assert notifier.claim_procesamiento(5) == "ok"
    assert llamadas[0][0] == "PATCH"
    assert llamadas[0][1] == "http://api/api/procesamientos/5/iniciar"


def test_claim_409_es_conflict_y_no_se_reintenta(notifier, monkeypatch):
    llamadas = patch_requests(monkeypatch, [FakeResponse(409)])
    assert notifier.claim_procesamiento(5) == "conflict"
    assert len(llamadas) == 1


def test_claim_404_es_error_y_no_se_reintenta(notifier, monkeypatch):
    llamadas = patch_requests(monkeypatch, [FakeResponse(404)])
    assert notifier.claim_procesamiento(5) == "error"
    assert len(llamadas) == 1


def test_claim_con_error_de_red_reintenta_y_devuelve_error(notifier, monkeypatch):
    llamadas = patch_requests(monkeypatch, [requests.exceptions.ConnectionError("caído")] * 3)
    assert notifier.claim_procesamiento(5) == "error"
    assert len(llamadas) == 3


@pytest.mark.parametrize("status", [500, 503, 429, 408])
def test_reintenta_5xx_408_y_429(notifier, monkeypatch, status):
    llamadas = patch_requests(monkeypatch, [FakeResponse(status)] * 3)
    assert notifier.claim_procesamiento(5) == "error"
    assert len(llamadas) == 3


@pytest.mark.parametrize("status", [400, 401, 403, 422])
def test_no_reintenta_otros_4xx(notifier, monkeypatch, status):
    llamadas = patch_requests(monkeypatch, [FakeResponse(status)])
    assert notifier.claim_procesamiento(5) == "error"
    assert len(llamadas) == 1


def test_reintenta_y_termina_ok_si_la_segunda_funciona(notifier, monkeypatch):
    llamadas = patch_requests(monkeypatch, [FakeResponse(500), FakeResponse(200)])
    assert notifier.claim_procesamiento(5) == "ok"
    assert len(llamadas) == 2


def test_get_pending_devuelve_la_lista(notifier, monkeypatch):
    patch_requests(monkeypatch, [FakeResponse(200, [{"id_procesamiento": 1}])])
    assert notifier.get_pending() == [{"id_procesamiento": 1}]


def test_get_pending_devuelve_none_si_falla_o_no_es_lista(notifier, monkeypatch):
    patch_requests(monkeypatch, [FakeResponse(200, {"x": 1})])
    assert notifier.get_pending() is None
    patch_requests(monkeypatch, [FakeResponse(401)])
    assert notifier.get_pending() is None


def test_sync_videos_envia_la_lista_completa(notifier, monkeypatch):
    llamadas = patch_requests(monkeypatch, [FakeResponse(200, {"registrados": 1, "actualizados": 0, "no_disponibles": 0})])
    videos = [{"nombre_archivo": "a.mp4", "tamano_bytes": 10, "fecha_modificacion": "2026-09-20T15:02:11"}]
    assert notifier.sync_videos(videos) is True
    assert llamadas[0][2]["json"] == {"videos": videos}


def test_sync_videos_false_si_falla(notifier, monkeypatch):
    patch_requests(monkeypatch, [FakeResponse(422)])
    assert notifier.sync_videos([]) is False