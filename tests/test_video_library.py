from datetime import datetime

import pytest

from src.video_library import VideoLibraryError, resolve_safe, scan

EXTS = [".mp4", ".avi"]


@pytest.fixture
def library(tmp_path):
    root = tmp_path / "videos"
    (root / "plaza").mkdir(parents=True)
    (root / "a.mp4").write_bytes(b"x" * 10)
    (root / "plaza" / "b.mp4").write_bytes(b"y" * 20)
    (root / "nota.txt").write_text("no es video")
    (tmp_path / "x.mp4").write_bytes(b"z")  # archivo FUERA de la carpeta
    return root


# ---------------- scan ----------------

def test_scan_filtra_por_extension(library):
    assert [v["nombre_archivo"] for v in scan(library, EXTS)] == ["a.mp4"]


def test_scan_recursivo_usa_rutas_relativas_con_barra(library):
    nombres = [v["nombre_archivo"] for v in scan(library, EXTS, recursive=True)]
    assert nombres == ["a.mp4", "plaza/b.mp4"]


def test_scan_acepta_extensiones_sin_punto_y_en_mayusculas(library):
    (library / "C.MP4").write_bytes(b"c")
    nombres = [v["nombre_archivo"] for v in scan(library, ["mp4"])]
    assert "C.MP4" in nombres


def test_scan_devuelve_tamano_y_fecha_sin_zona_horaria(library):
    video = scan(library, EXTS)[0]
    assert video["tamano_bytes"] == 10
    assert datetime.fromisoformat(video["fecha_modificacion"]).tzinfo is None


def test_scan_carpeta_inexistente_lanza_error(tmp_path):
    with pytest.raises(VideoLibraryError):
        scan(tmp_path / "no_existe", EXTS)


def test_scan_carpeta_vacia_es_valida(tmp_path):
    vacia = tmp_path / "vacia"
    vacia.mkdir()
    assert scan(vacia, EXTS) == []


def test_scan_no_lista_symlink_hacia_fuera(library, tmp_path):
    try:
        (library / "enlace.mp4").symlink_to(tmp_path / "x.mp4")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks no disponibles en este sistema")
    assert "enlace.mp4" not in [v["nombre_archivo"] for v in scan(library, EXTS)]


# ---------------- resolve_safe ----------------

def test_resolve_safe_archivo_valido(library):
    assert resolve_safe(library, "a.mp4", EXTS) == (library / "a.mp4").resolve()


def test_resolve_safe_archivo_en_subcarpeta(library):
    assert resolve_safe(library, "plaza/b.mp4", EXTS) == (library / "plaza" / "b.mp4").resolve()


@pytest.mark.parametrize("nombre", [
    "../x.mp4",
    "plaza/../../x.mp4",
    "/etc/passwd",
    "/a.mp4",
    "C:\\Windows\\x.mp4",
    "C:x.mp4",
    "plaza\\b.mp4",
    "a.mp4\x00.txt",
    "",
    None,
])
def test_resolve_safe_rechaza_nombres_peligrosos(library, nombre):
    assert resolve_safe(library, nombre, EXTS) is None


def test_resolve_safe_rechaza_archivo_inexistente(library):
    assert resolve_safe(library, "falta.mp4", EXTS) is None


def test_resolve_safe_rechaza_extension_no_permitida(library):
    assert resolve_safe(library, "nota.txt", EXTS) is None


def test_resolve_safe_rechaza_directorios(library):
    assert resolve_safe(library, "plaza", EXTS) is None


def test_resolve_safe_rechaza_symlink_hacia_fuera(library, tmp_path):
    try:
        (library / "enlace.mp4").symlink_to(tmp_path / "x.mp4")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks no disponibles en este sistema")
    assert resolve_safe(library, "enlace.mp4", EXTS) is None