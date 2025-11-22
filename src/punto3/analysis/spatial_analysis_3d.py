"""
Análisis espacial de configuraciones óptimas optimizado con Numba para 3D.

Este módulo implementa funciones para analizar los patrones espaciales
de la distribución de Ti en la estructura 3D.
"""

import numpy as np
from numba import njit
from typing import Dict


@njit(fastmath=True, cache=True)
def calcular_distancias_Ti_Nd_3d(Ti_positions: np.ndarray, Nd_positions: np.ndarray) -> np.ndarray:
    """
    Calcula la distancia de cada Ti a su Nd más cercano en 3D.

    Args:
        Ti_positions: Array (8, 3) con posiciones de Ti
        Nd_positions: Array (16, 3) con posiciones de Nd

    Returns:
        Array (8,) con distancia de cada Ti a su Nd más cercano
    """
    n_Ti = len(Ti_positions)
    n_Nd = len(Nd_positions)

    distancias_min = np.zeros(n_Ti, dtype=np.float64)

    for i in range(n_Ti):
        ti_x = Ti_positions[i, 0]
        ti_y = Ti_positions[i, 1]
        ti_z = Ti_positions[i, 2]

        min_dist = 1e10  # Infinito

        for j in range(n_Nd):
            nd_x = Nd_positions[j, 0]
            nd_y = Nd_positions[j, 1]
            nd_z = Nd_positions[j, 2]

            dx = ti_x - nd_x
            dy = ti_y - nd_y
            dz = ti_z - nd_z
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)

            if dist < min_dist:
                min_dist = dist

        distancias_min[i] = min_dist

    return distancias_min


@njit(fastmath=True, cache=True)
def calcular_distancias_Ti_Ti_3d(Ti_positions: np.ndarray) -> np.ndarray:
    """
    Calcula las distancias entre todos los pares de Ti en 3D.

    Args:
        Ti_positions: Array (8, 3) con posiciones de Ti

    Returns:
        Array (28,) con todas las distancias Ti-Ti
        (28 = C(8,2) = 8*7/2 pares únicos)
    """
    n_Ti = len(Ti_positions)
    n_pairs = n_Ti * (n_Ti - 1) // 2

    distancias = np.zeros(n_pairs, dtype=np.float64)
    idx = 0

    for i in range(n_Ti):
        for j in range(i + 1, n_Ti):
            dx = Ti_positions[i, 0] - Ti_positions[j, 0]
            dy = Ti_positions[i, 1] - Ti_positions[j, 1]
            dz = Ti_positions[i, 2] - Ti_positions[j, 2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)

            distancias[idx] = dist
            idx += 1

    return distancias


@njit(fastmath=True, cache=True)
def calcular_centro_de_masa_3d(positions: np.ndarray) -> np.ndarray:
    """
    Calcula el centro de masa de un conjunto de posiciones en 3D.

    Args:
        positions: Array (N, 3) con posiciones

    Returns:
        Array (3,) con centro de masa (x, y, z)
    """
    return np.mean(positions, axis=0)


def analizar_patron_espacial_3d(
    all_positions: np.ndarray,
    atom_types: np.ndarray,
    Ti_indices: np.ndarray,
    Nd_positions: np.ndarray
) -> Dict:
    """
    Analiza el patrón espacial de la distribución de Ti en 3D.

    Args:
        all_positions: Array (112, 3) con todas las coordenadas
        atom_types: Array (112,) con tipos de átomos
        Ti_indices: Array (8,) con índices de Ti
        Nd_positions: Array (16, 3) con posiciones de Nd

    Returns:
        Dict con métricas espaciales:
            - 'dist_Ti_Nd_promedio': Distancia promedio Ti → Nd más cercano
            - 'dist_Ti_Nd_min': Distancia mínima Ti → Nd
            - 'dist_Ti_Nd_max': Distancia máxima Ti → Nd
            - 'dist_Ti_Ti_promedio': Distancia promedio entre Ti
            - 'dist_Ti_Ti_min': Distancia mínima entre Ti
            - 'dist_Ti_Ti_max': Distancia máxima entre Ti
            - 'clustering_score': Métrica de agrupamiento (0-1, mayor = más agrupado)
            - 'dist_centro_promedio': Distancia promedio de Ti al centro de masa
    """
    # Extraer posiciones de Ti
    Ti_positions = all_positions[Ti_indices]

    # Calcular distancias Ti-Nd
    dists_Ti_Nd = calcular_distancias_Ti_Nd_3d(Ti_positions, Nd_positions)

    # Calcular distancias Ti-Ti
    dists_Ti_Ti = calcular_distancias_Ti_Ti_3d(Ti_positions)

    # Centro de masa de todos los átomos
    centro_sistema = calcular_centro_de_masa_3d(all_positions)

    # Distancias de Ti al centro
    dists_al_centro = np.array([
        np.linalg.norm(Ti_positions[i] - centro_sistema)
        for i in range(len(Ti_positions))
    ])

    # Clustering score: ratio entre dist Ti-Ti promedio y la máxima teórica
    # Para 112 átomos en ~17 Å × 17 Å × 2.4 Å, dist_max ≈ 23 Å
    dist_max_teorica = np.sqrt(17.2**2 + 17.2**2 + 2.4**2)
    clustering_score = 1.0 - (np.mean(dists_Ti_Ti) / dist_max_teorica)

    return {
        'dist_Ti_Nd_promedio': float(np.mean(dists_Ti_Nd)),
        'dist_Ti_Nd_min': float(np.min(dists_Ti_Nd)),
        'dist_Ti_Nd_max': float(np.max(dists_Ti_Nd)),
        'dist_Ti_Ti_promedio': float(np.mean(dists_Ti_Ti)),
        'dist_Ti_Ti_min': float(np.min(dists_Ti_Ti)),
        'dist_Ti_Ti_max': float(np.max(dists_Ti_Ti)),
        'clustering_score': float(clustering_score),
        'dist_centro_promedio': float(np.mean(dists_al_centro)),
        'Ti_positions': Ti_positions  # Guardar para visualización
    }


def interpretar_patron_3d(patron: Dict) -> str:
    """
    Interpreta las métricas espaciales y genera un resumen textual.

    Args:
        patron: Diccionario de métricas (salida de analizar_patron_espacial_3d)

    Returns:
        String con interpretación del patrón

    Examples:
        >>> patron = {'dist_Ti_Nd_promedio': 5.2, 'clustering_score': 0.65, ...}
        >>> print(interpretar_patron_3d(patron))
        Los átomos de Ti se ubican a distancia MODERADA de los átomos de Nd...
    """
    dist_Ti_Nd = patron['dist_Ti_Nd_promedio']
    dist_Ti_Ti = patron['dist_Ti_Ti_promedio']
    clustering = patron['clustering_score']

    interpretacion = []

    # Análisis Ti-Nd
    if dist_Ti_Nd < 3.0:
        interpretacion.append(f"Los átomos de Ti se ubican MUY CERCA de los átomos de Nd (distancia promedio: {dist_Ti_Nd:.2f} Å).")
    elif dist_Ti_Nd < 6.0:
        interpretacion.append(f"Los átomos de Ti se ubican a distancia MODERADA de los átomos de Nd (distancia promedio: {dist_Ti_Nd:.2f} Å).")
    else:
        interpretacion.append(f"Los átomos de Ti se ubican LEJOS de los átomos de Nd (distancia promedio: {dist_Ti_Nd:.2f} Å).")

    # Análisis clustering
    if clustering > 0.7:
        interpretacion.append(f"Los Ti tienden a AGRUPARSE fuertemente (clustering score: {clustering:.3f}, dist. promedio Ti-Ti: {dist_Ti_Ti:.2f} Å).")
    elif clustering > 0.5:
        interpretacion.append(f"Los Ti tienden a AGRUPARSE moderadamente (clustering score: {clustering:.3f}, dist. promedio Ti-Ti: {dist_Ti_Ti:.2f} Å).")
    else:
        interpretacion.append(f"Los Ti se DISPERSAN en la estructura (clustering score: {clustering:.3f}, dist. promedio Ti-Ti: {dist_Ti_Ti:.2f} Å).")

    return "\n".join(interpretacion)
