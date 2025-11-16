"""
Búsqueda exhaustiva (fuerza bruta) para optimización de configuración.

Para el Problema 1, con solo 12 posiciones candidatas, podemos evaluar
todas las configuraciones posibles y encontrar el óptimo global exacto.
"""

import numpy as np
from typing import Tuple, Dict
from ..grid import Grid2D, compute_total_energy


def brute_force_search(grid: Grid2D) -> Dict:
    """
    Encuentra la posición óptima del Ti mediante búsqueda exhaustiva.

    Evalúa todas las posiciones posibles para el átomo de Ti y retorna
    la que minimiza la energía total del sistema.

    Args:
        grid: Grilla 2D inicializada (sin Ti asignado, o se ignorará)

    Returns:
        Diccionario con:
            - 'best_position_idx': Índice óptimo en Fe_positions
            - 'best_energy': Energía mínima encontrada
            - 'all_energies': Array con todas las energías evaluadas
            - 'n_evaluations': Número de configuraciones evaluadas

    Examples:
        >>> from src.grid import create_grid_4x4
        >>> grid = create_grid_4x4()
        >>> result = brute_force_search(grid)
        >>> result['n_evaluations']
        12
        >>> 0 <= result['best_position_idx'] < 12
        True

    Note:
        Este método es factible solo para espacios pequeños.
        Para el Problema 2 (10×10 con 8 Ti), el espacio tiene ~1.5×10^10
        configuraciones, haciendo inviable la búsqueda exhaustiva.
    """
    n_positions = grid.n_Fe_sites
    all_energies = np.zeros(n_positions)

    best_idx = 0
    best_energy = np.inf

    # Evaluar cada posición posible
    for idx in range(n_positions):
        # Crear copia y asignar Ti
        test_grid = grid.copy()
        test_grid.set_Ti_position(idx)

        # Calcular energía
        energy = compute_total_energy(test_grid)
        all_energies[idx] = energy

        # Actualizar mejor solución
        if energy < best_energy:
            best_energy = energy
            best_idx = idx

    return {
        'best_position_idx': best_idx,
        'best_energy': best_energy,
        'all_energies': all_energies,
        'n_evaluations': n_positions
    }


def get_top_k_positions(grid: Grid2D, k: int = 3) -> Dict:
    """
    Encuentra las k mejores posiciones para el Ti.

    Útil para análisis y comparación con resultados de SA.

    Args:
        grid: Grilla 2D
        k: Número de mejores posiciones a retornar

    Returns:
        Diccionario con:
            - 'positions': Índices de las k mejores posiciones (ordenadas)
            - 'energies': Energías correspondientes
            - 'energy_gaps': Diferencias con respecto al óptimo

    Examples:
        >>> from src.grid import create_grid_4x4
        >>> grid = create_grid_4x4()
        >>> top3 = get_top_k_positions(grid, k=3)
        >>> len(top3['positions'])
        3
    """
    result = brute_force_search(grid)
    all_energies = result['all_energies']

    # Obtener índices ordenados por energía
    sorted_indices = np.argsort(all_energies)
    top_k_indices = sorted_indices[:k]

    top_k_energies = all_energies[top_k_indices]
    energy_gaps = top_k_energies - result['best_energy']

    return {
        'positions': top_k_indices.tolist(),
        'energies': top_k_energies.tolist(),
        'energy_gaps': energy_gaps.tolist()
    }


def analyze_energy_landscape(grid: Grid2D) -> Dict:
    """
    Analiza el paisaje de energía completo del problema.

    Calcula estadísticas útiles para entender la dificultad del problema
    y diseñar parámetros de Simulated Annealing.

    Args:
        grid: Grilla 2D

    Returns:
        Diccionario con estadísticas:
            - 'min_energy': Energía mínima (óptimo global)
            - 'max_energy': Energía máxima
            - 'mean_energy': Energía promedio
            - 'std_energy': Desviación estándar
            - 'energy_range': Rango de energías
            - 'n_local_minima': Número de mínimos locales (aproximado)

    Examples:
        >>> from src.grid import create_grid_4x4
        >>> grid = create_grid_4x4()
        >>> landscape = analyze_energy_landscape(grid)
        >>> landscape['min_energy'] < landscape['mean_energy']
        True
    """
    result = brute_force_search(grid)
    energies = result['all_energies']

    # Detectar mínimos locales (aproximado)
    # Un punto es mínimo local si es menor que sus "vecinos"
    # Para este problema simple, contamos configuraciones que son mejores
    # que el 50% de las otras
    median_energy = np.median(energies)
    n_below_median = np.sum(energies < median_energy)

    return {
        'min_energy': float(np.min(energies)),
        'max_energy': float(np.max(energies)),
        'mean_energy': float(np.mean(energies)),
        'std_energy': float(np.std(energies)),
        'energy_range': float(np.max(energies) - np.min(energies)),
        'n_local_minima': int(n_below_median)
    }
