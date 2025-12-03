"""
Análisis espacial de configuraciones óptimas optimizado con Numba.

Este módulo implementa funciones para analizar los patrones espaciales
de la distribución de Ti en la grilla, necesarias para responder las
preguntas del Punto 3:
1. ¿Dónde se ubican los Ti?
2. ¿Se agrupan o dispersan?
3. ¿Confirma la hipótesis del Punto 1?

IMPORTANTE: Las posiciones están en coordenadas físicas (Angstroms) con
espaciado de 2.8 Å entre átomos adyacentes.
"""

import numpy as np
from numba import njit
from typing import Dict, Tuple

# Constante de espaciado de la grilla (Angstroms)
GRID_SPACING = 2.8


@njit(fastmath=True, cache=True)
def calcular_distancias_Ti_Nd(Ti_positions: np.ndarray, Nd_positions: np.ndarray) -> np.ndarray:
    """
    Calcula la distancia de cada Ti a su Nd más cercano.

    Esta métrica ayuda a responder: "¿Los Ti se alejan de los Nd?"

    Args:
        Ti_positions: Array (8, 2) con posiciones de Ti
        Nd_positions: Array (16, 2) con posiciones de Nd

    Returns:
        Array (8,) con distancia de cada Ti a su Nd más cercano

    Examples:
        >>> Ti_pos = np.array([[0, 0], [9, 9]], dtype=np.int8)
        >>> Nd_pos = np.array([[3, 3], [4, 4]], dtype=np.int8)
        >>> dists = calcular_distancias_Ti_Nd(Ti_pos, Nd_pos)
        >>> dists.shape
        (2,)
    """
    n_Ti = len(Ti_positions)
    n_Nd = len(Nd_positions)

    distancias_min = np.zeros(n_Ti, dtype=np.float64)

    for i in range(n_Ti):
        ti_x = float(Ti_positions[i, 0])
        ti_y = float(Ti_positions[i, 1])

        min_dist = 1e10  # Infinito

        for j in range(n_Nd):
            nd_x = float(Nd_positions[j, 0])
            nd_y = float(Nd_positions[j, 1])

            dx = ti_x - nd_x
            dy = ti_y - nd_y
            dist = np.sqrt(dx * dx + dy * dy)

            if dist < min_dist:
                min_dist = dist

        distancias_min[i] = min_dist

    return distancias_min


@njit(fastmath=True, cache=True)
def calcular_distancias_Ti_Ti(Ti_positions: np.ndarray) -> np.ndarray:
    """
    Calcula las distancias entre todos los pares de Ti.

    Esta métrica ayuda a responder: "¿Los Ti se agrupan o dispersan?"

    Args:
        Ti_positions: Array (8, 2) con posiciones de Ti

    Returns:
        Array (28,) con todas las distancias Ti-Ti
        (28 = C(8,2) = 8*7/2 pares únicos)

    Examples:
        >>> Ti_pos = np.array([[0, 0], [3, 4], [1, 1]], dtype=np.int8)
        >>> dists = calcular_distancias_Ti_Ti(Ti_pos)
        >>> dists.shape
        (3,)  # C(3,2) = 3
    """
    n_Ti = len(Ti_positions)
    n_pairs = n_Ti * (n_Ti - 1) // 2

    distancias = np.zeros(n_pairs, dtype=np.float64)
    idx = 0

    for i in range(n_Ti):
        for j in range(i + 1, n_Ti):
            dx = float(Ti_positions[i, 0] - Ti_positions[j, 0])
            dy = float(Ti_positions[i, 1] - Ti_positions[j, 1])
            dist = np.sqrt(dx * dx + dy * dy)

            distancias[idx] = dist
            idx += 1

    return distancias


@njit(fastmath=True, cache=True)
def calcular_clustering_score(Ti_positions: np.ndarray, grid_size: int = 10) -> float:
    """
    Calcula un score de clustering (0 = máxima dispersión, 1 = máximo agrupamiento).

    El score se basa en la distancia promedio Ti-Ti normalizada:
    - Si los Ti están muy juntos → score alto (cercano a 1)
    - Si los Ti están dispersos → score bajo (cercano a 0)

    Args:
        Ti_positions: Array (8, 2) con posiciones de Ti en Angstroms
        grid_size: Tamaño de la grilla en índices (default: 10)

    Returns:
        Score de clustering en [0, 1]
        - 0: Dispersión máxima (Ti en esquinas opuestas)
        - 1: Agrupamiento máximo (todos los Ti juntos)

    Note:
        Este es un score simplificado. Valores bajos sugieren dispersión.

    Examples:
        >>> # Ti muy dispersos (coordenadas en Angstroms)
        >>> Ti_dispersos = np.array([[0,0], [25.2,25.2], [0,25.2], [25.2,0], [14,14], [5.6,5.6], [19.6,19.6], [2.8,22.4]])
        >>> score_disp = calcular_clustering_score(Ti_dispersos)
        >>> # Ti agrupados
        >>> Ti_agrupados = np.array([[0,0], [0,2.8], [2.8,0], [2.8,2.8], [5.6,0], [0,5.6], [5.6,2.8], [2.8,5.6]])
        >>> score_agrup = calcular_clustering_score(Ti_agrupados)
        >>> score_agrup > score_disp
        True
    """
    distancias_Ti_Ti = calcular_distancias_Ti_Ti(Ti_positions)

    # Distancia promedio entre Ti
    dist_promedio = np.mean(distancias_Ti_Ti)

    # Distancia máxima posible en la grilla (diagonal) en Angstroms
    dist_maxima = np.sqrt(2.0) * (grid_size - 1) * GRID_SPACING

    # Distancia mínima esperada si están dispersos (aproximación)
    # Si los 8 Ti se distribuyen uniformemente en una grilla 10×10
    dist_esperada_dispersa = dist_maxima / 2.0

    # Score: normalizar de forma que:
    # - dist_promedio pequeña → score alto (agrupados)
    # - dist_promedio grande → score bajo (dispersos)
    if dist_promedio > dist_esperada_dispersa:
        # Muy dispersos
        score = max(0.0, 1.0 - (dist_promedio / dist_maxima))
    else:
        # Relativamente agrupados
        score = 1.0 - (dist_promedio / dist_esperada_dispersa) * 0.5

    return score


@njit(fastmath=True, cache=True)
def calcular_distancia_al_centro(Ti_positions: np.ndarray, grid_size: int = 10) -> np.ndarray:
    """
    Calcula la distancia de cada Ti al centro de la grilla en Angstroms.

    El centro del núcleo Nd está en coordenadas físicas ((grid_size-1)/2 * GRID_SPACING) Å.
    Para grid_size=10 y GRID_SPACING=2.8: centro = (4.5 * 2.8, 4.5 * 2.8) = (12.6, 12.6) Å

    Args:
        Ti_positions: Array (8, 2) con posiciones de Ti en Angstroms
        grid_size: Tamaño de la grilla en índices (default: 10)

    Returns:
        Array (8,) con distancia de cada Ti al centro en Angstroms

    Examples:
        >>> Ti_pos = np.array([[0.0, 0.0], [25.2, 25.2]], dtype=np.float32)
        >>> dists = calcular_distancia_al_centro(Ti_pos)
        >>> dists[0]  # Ti en (0,0) está más lejos del centro que Ti en (25.2, 25.2)
        17.8...
    """
    n_Ti = len(Ti_positions)
    # Centro en coordenadas físicas (Angstroms)
    centro_x = (grid_size - 1) / 2.0 * GRID_SPACING
    centro_y = (grid_size - 1) / 2.0 * GRID_SPACING

    distancias = np.zeros(n_Ti, dtype=np.float64)

    for i in range(n_Ti):
        dx = float(Ti_positions[i, 0]) - centro_x
        dy = float(Ti_positions[i, 1]) - centro_y
        dist = np.sqrt(dx * dx + dy * dy)
        distancias[i] = dist

    return distancias


def analizar_distribucion_radial(Ti_positions: np.ndarray, grid_size: int = 10) -> Dict[str, float]:
    """
    Analiza la distribución radial de los Ti respecto al centro de la grilla.

    Args:
        Ti_positions: Array (8, 2) con posiciones de Ti
        grid_size: Tamaño de la grilla

    Returns:
        Dict con métricas:
            - 'dist_centro_promedio': Distancia promedio al centro
            - 'dist_centro_min': Ti más cercano al centro
            - 'dist_centro_max': Ti más alejado del centro
            - 'dist_centro_std': Desviación estándar

    Examples:
        >>> Ti_pos = np.array([[0,0], [9,9], [0,9], [9,0]], dtype=np.int8)
        >>> stats = analizar_distribucion_radial(Ti_pos)
        >>> 'dist_centro_promedio' in stats
        True
    """
    distancias_centro = calcular_distancia_al_centro(Ti_positions, grid_size)

    return {
        'dist_centro_promedio': float(np.mean(distancias_centro)),
        'dist_centro_min': float(np.min(distancias_centro)),
        'dist_centro_max': float(np.max(distancias_centro)),
        'dist_centro_std': float(np.std(distancias_centro))
    }


def analizar_patron_espacial(
    Ti_positions: np.ndarray,
    Nd_positions: np.ndarray,
    grid_size: int = 10
) -> Dict[str, float]:
    """
    Análisis completo del patrón espacial de la configuración de Ti.

    Esta función calcula TODAS las métricas necesarias para responder
    las preguntas del Punto 3.

    Args:
        Ti_positions: Array (8, 2) con posiciones de Ti
        Nd_positions: Array (16, 2) con posiciones de Nd
        grid_size: Tamaño de la grilla

    Returns:
        Dict con todas las métricas:
            # Distancias Ti-Nd (pregunta: ¿Ti se aleja de Nd?)
            - 'dist_Ti_Nd_promedio': Distancia promedio Ti a Nd más cercano
            - 'dist_Ti_Nd_min': Mínima distancia Ti-Nd
            - 'dist_Ti_Nd_max': Máxima distancia Ti-Nd
            - 'dist_Ti_Nd_std': Desviación estándar

            # Distancias Ti-Ti (pregunta: ¿se agrupan o dispersan?)
            - 'dist_Ti_Ti_promedio': Distancia promedio entre Ti
            - 'dist_Ti_Ti_min': Par de Ti más cercanos
            - 'dist_Ti_Ti_max': Par de Ti más alejados
            - 'dist_Ti_Ti_std': Desviación estándar

            # Clustering
            - 'clustering_score': Score de agrupamiento (0=disperso, 1=agrupado)

            # Distribución radial
            - 'dist_centro_promedio': Distancia promedio al centro
            - 'dist_centro_max': Ti más alejado del centro

    Examples:
        >>> from src.punto2.grid import crear_grid_inicial, get_Nd_positions_fijas
        >>> grid, Ti_pos, Nd_pos = crear_grid_inicial(seed=42)
        >>> patron = analizar_patron_espacial(Ti_pos, Nd_pos)
        >>> print(f"Distancia promedio Ti-Nd: {patron['dist_Ti_Nd_promedio']:.3f}")
        >>> print(f"Clustering score: {patron['clustering_score']:.3f}")
    """
    # Calcular distancias Ti-Nd
    dists_Ti_Nd = calcular_distancias_Ti_Nd(Ti_positions, Nd_positions)

    # Calcular distancias Ti-Ti
    dists_Ti_Ti = calcular_distancias_Ti_Ti(Ti_positions)

    # Calcular clustering score
    clustering = calcular_clustering_score(Ti_positions, grid_size)

    # Calcular distribución radial
    dist_radial = analizar_distribucion_radial(Ti_positions, grid_size)

    # Consolidar resultados
    return {
        # Distancias Ti-Nd
        'dist_Ti_Nd_promedio': float(np.mean(dists_Ti_Nd)),
        'dist_Ti_Nd_min': float(np.min(dists_Ti_Nd)),
        'dist_Ti_Nd_max': float(np.max(dists_Ti_Nd)),
        'dist_Ti_Nd_std': float(np.std(dists_Ti_Nd)),

        # Distancias Ti-Ti
        'dist_Ti_Ti_promedio': float(np.mean(dists_Ti_Ti)),
        'dist_Ti_Ti_min': float(np.min(dists_Ti_Ti)),
        'dist_Ti_Ti_max': float(np.max(dists_Ti_Ti)),
        'dist_Ti_Ti_std': float(np.std(dists_Ti_Ti)),

        # Clustering
        'clustering_score': float(clustering),

        # Distribución radial
        'dist_centro_promedio': dist_radial['dist_centro_promedio'],
        'dist_centro_max': dist_radial['dist_centro_max']
    }


def interpretar_patron(patron: Dict[str, float]) -> str:
    """
    Genera una interpretación en texto del patrón espacial.

    Esta función ayuda a responder las preguntas del Punto 3 de forma automática.

    Args:
        patron: Dict retornado por analizar_patron_espacial()

    Returns:
        String con interpretación del patrón

    Examples:
        >>> patron = analizar_patron_espacial(Ti_pos, Nd_pos)
        >>> interpretacion = interpretar_patron(patron)
        >>> print(interpretacion)
        Los átomos de Ti tienden a ALEJARSE de los átomos de Nd...
    """
    interpretacion = []

    # Pregunta 1: ¿Dónde se ubican los Ti?
    dist_Ti_Nd = patron['dist_Ti_Nd_promedio']
    dist_centro = patron['dist_centro_promedio']

    # Umbrales actualizados para coordenadas en Angstroms (espaciado 2.8 Å)
    # 4.0 unidades de grilla → 11.2 Å
    # 5.0 unidades de grilla → 14.0 Å
    if dist_Ti_Nd > 11.2:
        interpretacion.append(
            f"Los átomos de Ti tienden a ALEJARSE de los átomos de Nd "
            f"(distancia promedio: {dist_Ti_Nd:.2f} Å)."
        )
    else:
        interpretacion.append(
            f"Los átomos de Ti se ubican relativamente CERCA de los átomos de Nd "
            f"(distancia promedio: {dist_Ti_Nd:.2f} Å)."
        )

    if dist_centro > 14.0:
        interpretacion.append(
            f"Los Ti se concentran en la PERIFERIA de la grilla "
            f"(distancia al centro: {dist_centro:.2f})."
        )
    else:
        interpretacion.append(
            f"Los Ti se distribuyen cerca del CENTRO de la grilla "
            f"(distancia al centro: {dist_centro:.2f})."
        )

    # Pregunta 2: ¿Se agrupan o dispersan?
    clustering = patron['clustering_score']
    dist_Ti_Ti = patron['dist_Ti_Ti_promedio']

    if clustering < 0.3:
        interpretacion.append(
            f"Los Ti tienden a DISPERSARSE en la grilla "
            f"(clustering score: {clustering:.3f}, dist. promedio Ti-Ti: {dist_Ti_Ti:.2f} Å)."
        )
    elif clustering > 0.7:
        interpretacion.append(
            f"Los Ti tienden a AGRUPARSE "
            f"(clustering score: {clustering:.3f}, dist. promedio Ti-Ti: {dist_Ti_Ti:.2f} Å)."
        )
    else:
        interpretacion.append(
            f"Los Ti muestran una distribución INTERMEDIA "
            f"(clustering score: {clustering:.3f}, dist. promedio Ti-Ti: {dist_Ti_Ti:.2f} Å)."
        )

    # Pregunta 3: ¿Confirma hipótesis?
    # Umbral actualizado: 4.0 unidades de grilla → 11.2 Å
    if dist_Ti_Nd > 11.2 and clustering < 0.5:
        interpretacion.append(
            "\n✓ Este patrón CONFIRMA la hipótesis del Punto 1: "
            "los átomos de Ti prefieren sitios alejados de los átomos de R (Nd), "
            "maximizando las distancias Ti-Nd y dispersándose en la grilla."
        )
    else:
        interpretacion.append(
            "\n✗ Este patrón NO confirma completamente la hipótesis del Punto 1."
        )

    return "\n".join(interpretacion)
