"""
Análisis espacial de configuraciones óptimas 3D optimizado con Numba.

Este módulo implementa funciones para analizar los patrones espaciales
de la distribución de Ti en la estructura cristalina 3D, necesarias para
responder las preguntas del Punto 3 y comparar con resultados de Skelland:

1. Distancia promedio entre cada Ti y su Nd más cercano
2. Distancia promedio entre cada Ti y su Ti más cercano
3. ¿Los Ti se dispersan o agrupan?
4. ¿Los Ti se alejan de los Nd?
"""

import numpy as np
from numba import njit
from typing import Dict


@njit(fastmath=True, cache=True)
def calcular_distancias_Ti_Nd_3d(Ti_positions: np.ndarray, Nd_positions: np.ndarray) -> np.ndarray:
    """
    Calcula la distancia de cada Ti a su Nd más cercano en 3D.

    Esta métrica ayuda a responder: "¿Los Ti se alejan de los Nd?"
    (Análisis cualitativo según Skelland)

    Args:
        Ti_positions: Array (8, 3) con posiciones 3D de Ti
        Nd_positions: Array (16, 3) con posiciones 3D de Nd

    Returns:
        Array (8,) con distancia de cada Ti a su Nd más cercano (en Angstroms)

    Examples:
        >>> Ti_pos = np.random.rand(8, 3)
        >>> Nd_pos = np.random.rand(16, 3)
        >>> dists = calcular_distancias_Ti_Nd_3d(Ti_pos, Nd_pos)
        >>> dists.shape
        (8,)
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

            # Distancia euclidiana 3D
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

    Esta métrica ayuda a responder: "¿Los Ti se agrupan o dispersan?"
    (Análisis cualitativo según Skelland: patrón óptimo maximiza distancias Ti-Ti)

    Args:
        Ti_positions: Array (8, 3) con posiciones 3D de Ti

    Returns:
        Array (28,) con todas las distancias Ti-Ti (en Angstroms)
        (28 = C(8,2) = 8*7/2 pares únicos)

    Examples:
        >>> Ti_pos = np.random.rand(8, 3)
        >>> dists = calcular_distancias_Ti_Ti_3d(Ti_pos)
        >>> dists.shape
        (28,)
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
def calcular_distancia_vecino_mas_cercano_Ti(Ti_positions: np.ndarray) -> np.ndarray:
    """
    Calcula la distancia de cada Ti a su Ti vecino más cercano.

    Métrica alternativa para analizar dispersión: si los vecinos más cercanos
    están lejos, los Ti están dispersos.

    Args:
        Ti_positions: Array (8, 3) con posiciones 3D de Ti

    Returns:
        Array (8,) con distancia de cada Ti a su Ti más cercano

    Examples:
        >>> Ti_pos = np.random.rand(8, 3)
        >>> dists = calcular_distancia_vecino_mas_cercano_Ti(Ti_pos)
        >>> dists.shape
        (8,)
    """
    n_Ti = len(Ti_positions)
    distancias_min = np.zeros(n_Ti, dtype=np.float64)

    for i in range(n_Ti):
        min_dist = 1e10

        for j in range(n_Ti):
            if i == j:
                continue

            dx = Ti_positions[i, 0] - Ti_positions[j, 0]
            dy = Ti_positions[i, 1] - Ti_positions[j, 1]
            dz = Ti_positions[i, 2] - Ti_positions[j, 2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)

            if dist < min_dist:
                min_dist = dist

        distancias_min[i] = min_dist

    return distancias_min


@njit(fastmath=True, cache=True)
def calcular_clustering_score_3d(Ti_positions: np.ndarray) -> float:
    """
    Calcula un score de clustering en 3D (0 = máxima dispersión, 1 = máximo agrupamiento).

    El score se basa en la distancia promedio Ti-Ti normalizada:
    - Si los Ti están muy juntos → score alto (cercano a 1)
    - Si los Ti están dispersos → score bajo (cercano a 0)

    Args:
        Ti_positions: Array (8, 3) con posiciones 3D de Ti

    Returns:
        Score de clustering en [0, 1]
        - 0: Dispersión máxima
        - 1: Agrupamiento máximo

    Note:
        En 3D no hay "grid_size" como en 2D. La normalización se hace
        respecto a la distancia máxima entre Ti observada.

    Examples:
        >>> Ti_pos = np.random.rand(8, 3) * 10  # Ti dispersos en caja 10×10×10
        >>> score = calcular_clustering_score_3d(Ti_pos)
        >>> 0.0 <= score <= 1.0
        True
    """
    distancias_Ti_Ti = calcular_distancias_Ti_Ti_3d(Ti_positions)

    # Distancia promedio entre Ti
    dist_promedio = np.mean(distancias_Ti_Ti)

    # Distancia máxima observada entre Ti
    dist_maxima = np.max(distancias_Ti_Ti)

    # Distancia mínima observada entre Ti
    dist_minima = np.min(distancias_Ti_Ti)

    # Score normalizado: 0 = disperso (dist_promedio cercana a max), 1 = agrupado (dist_promedio cercana a min)
    if dist_maxima > dist_minima:
        score = 1.0 - (dist_promedio - dist_minima) / (dist_maxima - dist_minima)
    else:
        score = 0.5  # Todos a la misma distancia (caso degenerado)

    return score


@njit(fastmath=True, cache=True)
def calcular_distancia_al_centroide_3d(Ti_positions: np.ndarray, all_positions: np.ndarray) -> np.ndarray:
    """
    Calcula la distancia de cada Ti al centroide del sistema 3D.

    El centroide es el centro de masa geométrico de todos los 112 átomos.

    Args:
        Ti_positions: Array (8, 3) con posiciones 3D de Ti
        all_positions: Array (112, 3) con posiciones 3D de todos los átomos

    Returns:
        Array (8,) con distancia de cada Ti al centroide

    Examples:
        >>> Ti_pos = np.random.rand(8, 3)
        >>> all_pos = np.random.rand(112, 3)
        >>> dists = calcular_distancia_al_centroide_3d(Ti_pos, all_pos)
        >>> dists.shape
        (8,)
    """
    n_Ti = len(Ti_positions)

    # Calcular centroide (promedio de todas las posiciones)
    centroide_x = np.mean(all_positions[:, 0])
    centroide_y = np.mean(all_positions[:, 1])
    centroide_z = np.mean(all_positions[:, 2])

    distancias = np.zeros(n_Ti, dtype=np.float64)

    for i in range(n_Ti):
        dx = Ti_positions[i, 0] - centroide_x
        dy = Ti_positions[i, 1] - centroide_y
        dz = Ti_positions[i, 2] - centroide_z
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        distancias[i] = dist

    return distancias


def analizar_patron_espacial_3d(
    all_positions: np.ndarray,
    Ti_indices: np.ndarray,
    Nd_start_idx: int = 96
) -> Dict[str, float]:
    """
    Análisis completo del patrón espacial 3D de la configuración de Ti.

    Esta función calcula TODAS las métricas necesarias para responder
    las preguntas del Punto 3 y comparar con resultados de Skelland.

    Args:
        all_positions: Array (112, 3) con posiciones 3D de todos los átomos
        Ti_indices: Array (8,) con índices (0-95) de átomos de Ti
        Nd_start_idx: Índice donde empiezan los Nd (default: 96)

    Returns:
        Dict con todas las métricas:
            # Distancias Ti-Nd (métrica de Skelland #1)
            - 'dist_Ti_Nd_promedio': Distancia promedio Ti a Nd más cercano (Å)
            - 'dist_Ti_Nd_min': Mínima distancia Ti-Nd (Å)
            - 'dist_Ti_Nd_max': Máxima distancia Ti-Nd (Å)
            - 'dist_Ti_Nd_std': Desviación estándar (Å)

            # Distancias Ti-Ti (métrica de Skelland #2)
            - 'dist_Ti_Ti_promedio': Distancia promedio entre Ti (Å)
            - 'dist_Ti_Ti_min': Par de Ti más cercanos (Å)
            - 'dist_Ti_Ti_max': Par de Ti más alejados (Å)
            - 'dist_Ti_Ti_std': Desviación estándar (Å)

            # Distancia al vecino Ti más cercano
            - 'dist_Ti_vecino_promedio': Distancia promedio al Ti más cercano (Å)
            - 'dist_Ti_vecino_min': Ti con vecino más cercano (Å)
            - 'dist_Ti_vecino_max': Ti con vecino más lejano (Å)

            # Clustering
            - 'clustering_score': Score de agrupamiento (0=disperso, 1=agrupado)

            # Distribución radial
            - 'dist_centroide_promedio': Distancia promedio al centroide (Å)
            - 'dist_centroide_max': Ti más alejado del centroide (Å)

    Examples:
        >>> patron = analizar_patron_espacial_3d(all_pos, Ti_idx)
        >>> print(f"Distancia promedio Ti-Nd: {patron['dist_Ti_Nd_promedio']:.3f} Å")
        >>> print(f"Distancia promedio Ti-Ti: {patron['dist_Ti_Ti_promedio']:.3f} Å")
        >>> print(f"Clustering score: {patron['clustering_score']:.3f}")
    """
    # Extraer posiciones de Ti y Nd
    Ti_positions = all_positions[Ti_indices]
    Nd_positions = all_positions[Nd_start_idx:Nd_start_idx+16]

    # Calcular distancias Ti-Nd
    dists_Ti_Nd = calcular_distancias_Ti_Nd_3d(Ti_positions, Nd_positions)

    # Calcular distancias Ti-Ti
    dists_Ti_Ti = calcular_distancias_Ti_Ti_3d(Ti_positions)

    # Calcular distancia al vecino Ti más cercano
    dists_Ti_vecino = calcular_distancia_vecino_mas_cercano_Ti(Ti_positions)

    # Calcular clustering score
    clustering = calcular_clustering_score_3d(Ti_positions)

    # Calcular distancias al centroide
    dists_centroide = calcular_distancia_al_centroide_3d(Ti_positions, all_positions)

    # Consolidar resultados
    return {
        # Distancias Ti-Nd (MÉTRICA CLAVE DE SKELLAND)
        'dist_Ti_Nd_promedio': float(np.mean(dists_Ti_Nd)),
        'dist_Ti_Nd_min': float(np.min(dists_Ti_Nd)),
        'dist_Ti_Nd_max': float(np.max(dists_Ti_Nd)),
        'dist_Ti_Nd_std': float(np.std(dists_Ti_Nd)),

        # Distancias Ti-Ti (MÉTRICA CLAVE DE SKELLAND)
        'dist_Ti_Ti_promedio': float(np.mean(dists_Ti_Ti)),
        'dist_Ti_Ti_min': float(np.min(dists_Ti_Ti)),
        'dist_Ti_Ti_max': float(np.max(dists_Ti_Ti)),
        'dist_Ti_Ti_std': float(np.std(dists_Ti_Ti)),

        # Distancia al vecino Ti más cercano
        'dist_Ti_vecino_promedio': float(np.mean(dists_Ti_vecino)),
        'dist_Ti_vecino_min': float(np.min(dists_Ti_vecino)),
        'dist_Ti_vecino_max': float(np.max(dists_Ti_vecino)),

        # Clustering
        'clustering_score': float(clustering),

        # Distribución radial
        'dist_centroide_promedio': float(np.mean(dists_centroide)),
        'dist_centroide_max': float(np.max(dists_centroide))
    }


def interpretar_patron_3d(patron: Dict[str, float]) -> str:
    """
    Genera una interpretación en texto del patrón espacial 3D.

    Compara con las conclusiones de Skelland:
    - Patrón óptimo maximiza distancia entre Ti
    - Patrón óptimo mantiene Ti alejados de Nd

    Args:
        patron: Dict retornado por analizar_patron_espacial_3d()

    Returns:
        String con interpretación del patrón

    Examples:
        >>> patron = analizar_patron_espacial_3d(all_pos, Ti_idx)
        >>> interpretacion = interpretar_patron_3d(patron)
        >>> print(interpretacion)
        Los átomos de Ti mantienen una distancia promedio de X.XX Å a los Nd...
    """
    interpretacion = []

    # Métrica 1: Distancia Ti-Nd
    dist_Ti_Nd = patron['dist_Ti_Nd_promedio']
    interpretacion.append(
        f"1. DISTANCIA Ti-Nd (métrica de Skelland):\n"
        f"   - Distancia promedio: {dist_Ti_Nd:.3f} Å\n"
        f"   - Rango: [{patron['dist_Ti_Nd_min']:.3f}, {patron['dist_Ti_Nd_max']:.3f}] Å\n"
        f"   - Desviación estándar: {patron['dist_Ti_Nd_std']:.3f} Å"
    )

    if dist_Ti_Nd > 5.0:
        interpretacion.append(
            f"   → Los Ti están ALEJADOS de los Nd (consistente con Skelland)"
        )
    else:
        interpretacion.append(
            f"   → Los Ti están relativamente CERCA de los Nd"
        )

    # Métrica 2: Distancia Ti-Ti
    dist_Ti_Ti = patron['dist_Ti_Ti_promedio']
    interpretacion.append(
        f"\n2. DISTANCIA Ti-Ti (métrica de Skelland):\n"
        f"   - Distancia promedio: {dist_Ti_Ti:.3f} Å\n"
        f"   - Rango: [{patron['dist_Ti_Ti_min']:.3f}, {patron['dist_Ti_Ti_max']:.3f}] Å\n"
        f"   - Desviación estándar: {patron['dist_Ti_Ti_std']:.3f} Å\n"
        f"   - Vecino Ti más cercano promedio: {patron['dist_Ti_vecino_promedio']:.3f} Å"
    )

    if dist_Ti_Ti > 7.0:
        interpretacion.append(
            f"   → Los Ti están DISPERSOS (consistente con Skelland: maximiza dist Ti-Ti)"
        )
    else:
        interpretacion.append(
            f"   → Los Ti están relativamente AGRUPADOS"
        )

    # Clustering
    clustering = patron['clustering_score']
    interpretacion.append(
        f"\n3. CLUSTERING:\n"
        f"   - Score: {clustering:.3f} (0=disperso, 1=agrupado)"
    )

    if clustering < 0.3:
        interpretacion.append(
            f"   → Distribución DISPERSA"
        )
    elif clustering > 0.7:
        interpretacion.append(
            f"   → Distribución AGRUPADA"
        )
    else:
        interpretacion.append(
            f"   → Distribución INTERMEDIA"
        )

    # Conclusión
    interpretacion.append("\n" + "="*60)
    if dist_Ti_Nd > 5.0 and dist_Ti_Ti > 7.0:
        interpretacion.append(
            "✓ CONSISTENTE con Skelland: Ti alejados de Nd y dispersos entre sí"
        )
    else:
        interpretacion.append(
            "✗ Patrón diferente a las conclusiones de Skelland"
        )
    interpretacion.append("="*60)

    return "\n".join(interpretacion)
