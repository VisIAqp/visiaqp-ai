from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from src.analyzer import ZoneAnalyzer
from src.zones import ZoneManager


def deteccion(id, x=25):
    return SimpleNamespace(track_id=id, confidence=0.8, center=(x, 25))


def test_solo_cuenta_ids_dentro_del_poligono_sin_repetir_frames():
    zona = ZoneManager.from_config({"id_zona": 1, "zone_polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]})
    analizador = ZoneAnalyzer(1)
    inicio = datetime(2026, 10, 31, 10)
    for segundo in (0, 30, 60):
        filtradas = zona.filter_detections_in_zone([deteccion(1), deteccion(2, 75)], 100, 100)
        analizador.add_sample(filtradas, inicio + timedelta(seconds=segundo))
    resultado = analizador.close_window(1)
    assert resultado.cantidad_peatones == 1
    assert resultado.confianza_promedio == pytest.approx(0.8)


def test_entrada_a_zona_y_reinicio_de_ventana():
    analizador = ZoneAnalyzer(1)
    inicio = datetime(2026, 10, 31, 10)
    analizador.add_sample([], inicio)
    analizador.add_sample([deteccion(7)], inicio + timedelta(seconds=30))
    assert analizador.close_window(1).cantidad_peatones == 1
    assert analizador.close_window(1) is None
    analizador.add_sample([], inicio + timedelta(seconds=60))
    assert analizador.close_window(1).cantidad_peatones == 0
    analizador.add_sample([deteccion(7)], inicio + timedelta(seconds=120))
    assert analizador.close_window(1).cantidad_peatones == 1


def test_ocupacion_ponderada_por_tiempo_no_por_numero_de_frames():
    analizador = ZoneAnalyzer(1)
    inicio = datetime(2026, 10, 31, 10)
    analizador.add_sample([deteccion(i) for i in range(10)], inicio)
    analizador.add_sample([deteccion(0), deteccion(1)], inicio + timedelta(seconds=10))
    analizador.add_sample([], inicio + timedelta(seconds=60))
    resultado = analizador.close_window(1)
    assert resultado.cantidad_peatones == 10
    assert resultado.promedio_presentes == pytest.approx(200 / 60)
    assert resultado.max_presentes == 10
    assert resultado.segundos_observados == 60
    assert resultado.to_dict()["promedio_presentes"] == resultado.promedio_presentes


def test_muestra_aislada_no_inventa_duracion_ni_ocupacion():
    analizador = ZoneAnalyzer(1)
    analizador.add_sample([deteccion(1)], datetime(2026, 10, 31, 10))
    resultado = analizador.close_window(1)
    assert resultado.cantidad_peatones == 1
    assert resultado.promedio_presentes is None
    assert resultado.segundos_observados is None


def test_rechaza_muestras_fuera_de_orden():
    analizador = ZoneAnalyzer(1)
    inicio = datetime(2026, 10, 31, 10)
    analizador.add_sample([], inicio)
    with pytest.raises(ValueError, match="orden temporal"):
        analizador.add_sample([], inicio - timedelta(seconds=1))
