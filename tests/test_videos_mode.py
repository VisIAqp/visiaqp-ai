import main

POLIGONO = [[0.05, 0.10], [0.95, 0.10], [0.95, 0.95], [0.05, 0.95]]


def _pendiente(id_proc, nombre_archivo, polygon=POLIGONO):
    return {
        "id_procesamiento": id_proc,
        "id_zona": 1,
        "id_camara": 1,
        "nombre_camara": "Zona de Prueba",
        "nombre_archivo": nombre_archivo,
        "fecha_captura_inicio": "2026-09-20T14:30:00",
        "zone_polygon": polygon,
    }


class FakeNotifier:
    """Imita al backend: un pendiente desaparece de la lista una vez que se intenta reclamar."""

    def __init__(self, pendientes, claims=None):
        self._todos = pendientes
        self._claims = claims or {}
        self.synced, self.claimed, self.closed = [], [], []

    def sync_videos(self, videos):
        self.synced.append(videos)
        return True

    def get_pending(self):
        return [p for p in self._todos if p["id_procesamiento"] not in self.claimed]

    def claim_procesamiento(self, id_procesamiento):
        self.claimed.append(id_procesamiento)
        return self._claims.get(id_procesamiento, "ok")

    def close_procesamiento(self, id_procesamiento, payload):
        self.closed.append((id_procesamiento, payload))
        return True


def _config(library_dir):
    return {
        "videos": {
            "library_dir": str(library_dir),
            "extensions": [".mp4"],
            "recursive": False,
            "catalog_sync_seconds": 60,
            "pending_poll_seconds": 0,
        },
        "logging": {"fallback_to_local_log": False},
    }


def _run(config, notifier, ciclos=3):
    main.run_videos_mode(config, notifier, max_cycles=ciclos, sleep=lambda s: None)


def test_procesa_el_valido_cierra_con_error_el_inexistente_y_omite_el_conflicto(tmp_path, monkeypatch):
    (tmp_path / "a.mp4").write_bytes(b"x")
    procesados = []
    monkeypatch.setattr(
        main, "process_camera",
        lambda camera_config, config, notifier, id_procesamiento_existente=None:
            procesados.append((camera_config, id_procesamiento_existente)),
    )

    notifier = FakeNotifier(
        [_pendiente(3, "a.mp4"), _pendiente(2, "falta.mp4"), _pendiente(1, "a.mp4")],
        claims={3: "conflict"},
    )
    _run(_config(tmp_path), notifier)

    assert notifier.claimed == [3, 2, 1]

    # El inexistente se reclama y se cierra con error (no queda pendiente para siempre)
    assert len(notifier.closed) == 1
    id_cerrado, payload = notifier.closed[0]
    assert id_cerrado == 2
    assert payload["estado"] == "error"
    assert "falta.mp4" in payload["mensaje_error"]

    # Solo se procesa el válido, con su procesamiento ya existente y la fecha de captura
    assert len(procesados) == 1
    camera_config, id_existente = procesados[0]
    assert id_existente == 1
    assert camera_config["source"] == str((tmp_path / "a.mp4").resolve())
    assert camera_config["event_start"] == "2026-09-20T14:30:00"
    assert camera_config["zone_polygon"] == POLIGONO


def test_pendiente_sin_poligono_se_cierra_con_error(tmp_path, monkeypatch):
    (tmp_path / "a.mp4").write_bytes(b"x")
    procesados = []
    monkeypatch.setattr(main, "process_camera", lambda *a, **k: procesados.append(1))

    notifier = FakeNotifier([_pendiente(1, "a.mp4", polygon=None)])
    _run(_config(tmp_path), notifier)

    assert procesados == []
    assert notifier.closed[0][1]["estado"] == "error"


def test_path_traversal_se_cierra_con_error_sin_procesar(tmp_path, monkeypatch):
    procesados = []
    monkeypatch.setattr(main, "process_camera", lambda *a, **k: procesados.append(1))

    notifier = FakeNotifier([_pendiente(1, "../secreto.mp4")])
    _run(_config(tmp_path), notifier)

    assert procesados == []
    assert notifier.closed[0][1]["estado"] == "error"


def test_error_de_red_al_reclamar_no_procesa_ni_cierra(tmp_path, monkeypatch):
    (tmp_path / "a.mp4").write_bytes(b"x")
    procesados = []
    monkeypatch.setattr(main, "process_camera", lambda *a, **k: procesados.append(1))

    notifier = FakeNotifier([_pendiente(1, "a.mp4")], claims={1: "error"})
    _run(_config(tmp_path), notifier)

    assert procesados == []
    assert notifier.closed == []


def test_catalogo_se_sincroniza_una_vez_por_intervalo(tmp_path, monkeypatch):
    (tmp_path / "a.mp4").write_bytes(b"x")
    notifier = FakeNotifier([])
    _run(_config(tmp_path), notifier, ciclos=3)

    assert len(notifier.synced) == 1
    assert [v["nombre_archivo"] for v in notifier.synced[0]] == ["a.mp4"]


def test_catalogo_no_se_sincroniza_si_la_carpeta_no_existe(tmp_path):
    notifier = FakeNotifier([])
    _run(_config(tmp_path / "no_existe"), notifier, ciclos=1)

    assert notifier.synced == []