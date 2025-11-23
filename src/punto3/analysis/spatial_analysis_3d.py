"""
Análisis espacial de configuraciones atómicas en 3D.

Este módulo provee funciones para analizar patrones espaciales en la configuración
optimizada de átomos de Ti en la estructura cristalina 3D.
"""

import numpy as np
from numba import njit
from typing import Dict


@njit(fastmath=True, cache=True)
def calcular_distancias_Ti_Nd_3d(Ti_positions: np.ndarray, Nd_positions: np.ndarray) -> np.ndarray:
    """
    Calcula la distancia mínima de cada Ti a su Nd más cercano.

    Args:
        Ti_positions: Array (n_Ti, 3) con posiciones de Ti
        Nd_positions: Array (16, 3) con posiciones de Nd

    Returns:
        Array (n_Ti,) con distancias mínimas Ti-Nd
    """
    n_Ti = len(Ti_positions)
    n_Nd = len(Nd_positions)
    distancias_min = np.zeros(n_Ti, dtype=np.float64)

    for i in range(n_Ti):
        ti_x, ti_y, ti_z = Ti_positions[i, 0], Ti_positions[i, 1], Ti_positions[i, 2]

        dist_min = np.inf
        for j in range(n_Nd):
            nd_x, nd_y, nd_z = Nd_positions[j, 0], Nd_positions[j, 1], Nd_positions[j, 2]

            # Distancia euclidiana 3D
            dx = ti_x - nd_x
            dy = ti_y - nd_y
            dz = ti_z - nd_z
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)

            if dist < dist_min:
                dist_min = dist

        distancias_min[i] = dist_min

    return distancias_min


@njit(fastmath=True, cache=True)
def calcular_distancias_Ti_Ti_3d(Ti_positions: np.ndarray) -> np.ndarray:
    """
    Calcula todas las distancias entre pares de átomos de Ti.

    Args:
        Ti_positions: Array (n_Ti, 3) con posiciones de Ti

    Returns:
        Array 1D con todas las distancias Ti-Ti (n_Ti * (n_Ti-1) / 2 elementos)
    """
    n_Ti = len(Ti_positions)
    n_pares = (n_Ti * (n_Ti - 1)) // 2

    distancias = np.zeros(n_pares, dtype=np.float64)

    idx = 0
    for i in range(n_Ti):
        ti1_x, ti1_y, ti1_z = Ti_positions[i, 0], Ti_positions[i, 1], Ti_positions[i, 2]

        for j in range(i + 1, n_Ti):
            ti2_x, ti2_y, ti2_z = Ti_positions[j, 0], Ti_positions[j, 1], Ti_positions[j, 2]

            # Distancia euclidiana 3D
            dx = ti1_x - ti2_x
            dy = ti1_y - ti2_y
            dz = ti1_z - ti2_z
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)

            distancias[idx] = dist
            idx += 1

    return distancias


@njit(fastmath=True, cache=True)
def calcular_centro_de_masa_3d(positions: np.ndarray) -> np.ndarray:
    """
    Calcula el centro de masa de un conjunto de posiciones 3D.

    IMPORTANTE: NO usar np.mean(axis=0) con Numba - causa TypingError.
    Implementación manual con loop.

    Args:
        positions: Array (N, 3) con posiciones

    Returns:
        Array (3,) con centro de masa (x, y, z)
    """
    n = len(positions)
    centro = np.zeros(3, dtype=np.float64)

    # Loop manual para evitar bug de Numba con axis=
    for i in range(n):
        centro[0] += positions[i, 0]
        centro[1] += positions[i, 1]
        centro[2] += positions[i, 2]

    centro[0] /= n
    centro[1] /= n
    centro[2] /= n

    return centro


def analizar_patron_espacial_3d(
    all_positions: np.ndarray,
    atom_types: np.ndarray,
    Ti_indices: np.ndarray,
    Nd_positions: np.ndarray
) -> Dict:
    """
    Analiza el patrón espacial de la configuración optimizada en 3D.

    Args:
        all_positions: Array (112, 3) con todas las posiciones
        atom_types: Array (112,) con tipos de átomos
        Ti_indices: Array (8,) con índices de Ti
        Nd_positions: Array (16, 3) con posiciones de Nd

    Returns:
        Diccionario con métricas espaciales:
        - dist_Ti_Nd_promedio: Distancia promedio Ti-Nd
        - dist_Ti_Ti_promedio: Distancia promedio Ti-Ti
        - clustering_score: Métrica de agrupamiento (0-1)
        - dist_centro_promedio: Distancia promedio de Ti al centro
        - dist_Ti_Nd_all: Array con todas las distancias Ti-Nd
        - dist_Ti_Ti_all: Array con todas las distancias Ti-Ti
    """
    # Extraer posiciones de Ti
    Ti_positions = all_positions[Ti_indices]

    # Distancias Ti-Nd (mínima de cada Ti a Nd más cercano)
    dists_Ti_Nd = calcular_distancias_Ti_Nd_3d(Ti_positions, Nd_positions)

    # Distancias Ti-Ti (todas las pares)
    dists_Ti_Ti = calcular_distancias_Ti_Ti_3d(Ti_positions)

    # Centro de masa de todos los átomos
    centro_sistema = calcular_centro_de_masa_3d(all_positions)

    # Distancias de Ti al centro
    dists_al_centro = np.array([
        np.linalg.norm(Ti_positions[i] - centro_sistema)
        for i in range(len(Ti_positions))
    ])

    # Clustering score: inversa de la distancia promedio Ti-Ti normalizada
    # Valores altos → Ti agrupados, valores bajos → Ti dispersos
    # Normalizar por la distancia máxima teórica del sistema
    dist_max_teorica = np.linalg.norm(all_positions.max(axis=0) - all_positions.min(axis=0))
    clustering_score = 1.0 - (np.mean(dists_Ti_Ti) / dist_max_teorica)

    return {
        'dist_Ti_Nd_promedio': float(np.mean(dists_Ti_Nd)),
        'dist_Ti_Ti_promedio': float(np.mean(dists_Ti_Ti)),
        'clustering_score': float(clustering_score),
        'dist_centro_promedio': float(np.mean(dists_al_centro)),
        'dist_Ti_Nd_all': dists_Ti_Nd,
        'dist_Ti_Ti_all': dists_Ti_Ti,
        'dist_Ti_Nd_std': float(np.std(dists_Ti_Nd)),
        'dist_Ti_Ti_std': float(np.std(dists_Ti_Ti))
    }


def interpretar_patron_3d(patron: Dict) -> str:
    """
    Genera interpretación en lenguaje natural del patrón espacial.

    Args:
        patron: Diccionario retornado por analizar_patron_espacial_3d

    Returns:
        String con interpretación textual
    """
    dist_Ti_Nd = patron['dist_Ti_Nd_promedio']
    dist_Ti_Ti = patron['dist_Ti_Ti_promedio']
    clustering = patron['clustering_score']
    dist_centro = patron['dist_centro_promedio']

    interpretacion = []

    # Análisis Ti-Nd
    if dist_Ti_Nd < 3.0:
        interpretacion.append(
            f"Los átomos de Ti se ubican MUY CERCA de los átomos de Nd "
            f"(distancia promedio: {dist_Ti_Nd:.3f} Å)."
        )
    elif dist_Ti_Nd < 5.0:
        interpretacion.append(
            f"Los átomos de Ti se ubican relativamente CERCA de los átomos de Nd "
            f"(distancia promedio: {dist_Ti_Nd:.3f} Å)."
        )
    else:
        interpretacion.append(
            f"Los átomos de Ti se ubican LEJOS de los átomos de Nd "
            f"(distancia promedio: {dist_Ti_Nd:.3f} Å)."
        )

    # Análisis distribución espacial
    if dist_centro < 5.0:
        interpretacion.append(
            f"Los Ti se concentran cerca del CENTRO de la estructura cristalina "
            f"(distancia al centro: {dist_centro:.3f} Å)."
        )
    else:
        interpretacion.append(
            f"Los Ti se distribuyen alejados del centro "
            f"(distancia al centro: {dist_centro:.3f} Å)."
        )

    # Análisis clustering
    if clustering > 0.7:
        interpretacion.append(
            f"Los Ti tienden a AGRUPARSE fuertemente "
            f"(clustering score: {clustering:.3f}, dist. promedio Ti-Ti: {dist_Ti_Ti:.3f} Å)."
        )
    elif clustering > 0.4:
        interpretacion.append(
            f"Los Ti muestran agrupamiento MODERADO "
            f"(clustering score: {clustering:.3f}, dist. promedio Ti-Ti: {dist_Ti_Ti:.3f} Å)."
        )
    else:
        interpretacion.append(
            f"Los Ti están DISPERSOS en la estructura "
            f"(clustering score: {clustering:.3f}, dist. promedio Ti-Ti: {dist_Ti_Ti:.3f} Å)."
        )

    return "\n".join(interpretacion)
