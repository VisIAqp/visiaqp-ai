"""
video_library.py
Catálogo de videos pregrabados en una carpeta (local, disco compartido
o NAS montado). No conoce la API ni la detección.

Seguridad: el nombre de archivo de un procesamiento llega desde la API,
así que NUNCA se concatena directo a una ruta. resolve_safe() es el único
punto autorizado para convertir un nombre en una ruta que el motor abre.
"""

from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable

from loguru import logger

MAX_NOMBRE_LEN = 255


class VideoLibraryError(Exception):
    """La carpeta de videos no existe o no es accesible."""


def _normalize_extensions(extensions: Iterable[str]) -> set[str]:
    """Normaliza a minúsculas y con punto inicial: 'MP4' -> '.mp4'."""
    return {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}


def _is_safe_relative_name(nombre) -> bool:
    """
    Un nombre válido es una ruta RELATIVA, con '/' como separador, sin
    '..', sin raíz ni letra de unidad, sin '\\' ni NUL, y de hasta 255
    caracteres.
    """
    if not isinstance(nombre, str) or not nombre or len(nombre) > MAX_NOMBRE_LEN:
        return False
    if "\x00" in nombre or "\\" in nombre:
        return False
    if PurePosixPath(nombre).anchor or PureWindowsPath(nombre).anchor:
        return False  # absoluta, o con unidad tipo "C:"
    if ".." in PurePosixPath(nombre).parts:
        return False
    return True


def scan(library_dir: str, extensions: Iterable[str], recursive: bool = False) -> list[dict]:
    """
    Lista los videos de la carpeta. Cada elemento:
      {"nombre_archivo": ruta relativa con '/', "tamano_bytes": int,
       "fecha_modificacion": ISO 8601 sin zona horaria}

    Solo devuelve archivos que resolve_safe() aceptaría: se omiten los
    enlaces simbólicos que apunten fuera de la carpeta.

    Lanza VideoLibraryError si la carpeta no existe o no es accesible,
    para no confundirla con una carpeta vacía (que marcaría todo el
    catálogo como no disponible).
    """
    root = Path(library_dir)
    if not root.is_dir():
        raise VideoLibraryError(f"La carpeta de videos no existe o no es accesible: {library_dir}")

    exts = _normalize_extensions(extensions)
    root_resolved = root.resolve()
    candidates = root.rglob("*") if recursive else root.glob("*")

    videos: list[dict] = []
    try:
        for path in candidates:
            if path.suffix.lower() not in exts:
                continue
            try:
                if not path.is_file():
                    continue
                resolved = path.resolve()
                if not resolved.is_relative_to(root_resolved):
                    logger.warning(f"Se omite {path}: apunta fuera de la carpeta de videos")
                    continue
                stat = resolved.stat()
            except OSError as e:
                logger.warning(f"Se omite {path}: no se pudo leer ({e})")
                continue

            nombre = path.relative_to(root).as_posix()
            if not _is_safe_relative_name(nombre):
                logger.warning(f"Se omite un archivo con nombre no válido para el catálogo: {nombre!r}")
                continue

            videos.append({
                "nombre_archivo": nombre,
                "tamano_bytes": stat.st_size,
                "fecha_modificacion": datetime.fromtimestamp(stat.st_mtime)
                .replace(microsecond=0)
                .isoformat(),
            })
    except OSError as e:
        raise VideoLibraryError(f"No se pudo recorrer la carpeta de videos: {e}") from e

    videos.sort(key=lambda v: v["nombre_archivo"])
    return videos


def resolve_safe(library_dir: str, nombre_archivo, extensions: Iterable[str]) -> Path | None:
    """
    Convierte un nombre de archivo (venido de la API) en una ruta absoluta,
    solo si es seguro. Devuelve None si el nombre es absoluto o con '..',
    si la ruta resuelta queda fuera de library_dir (p. ej. un symlink),
    si el archivo no existe, o si la extensión no está permitida.
    """
    if not _is_safe_relative_name(nombre_archivo):
        return None
    if Path(nombre_archivo).suffix.lower() not in _normalize_extensions(extensions):
        return None

    try:
        root = Path(library_dir).resolve()
        candidate = (root / nombre_archivo).resolve()
        if not candidate.is_relative_to(root):
            return None
        if not candidate.is_file():
            return None
    except (OSError, ValueError, RuntimeError):  # RuntimeError: bucles de symlinks
        return None

    return candidate