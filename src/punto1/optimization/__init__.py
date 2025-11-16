"""
Módulo optimization: Algoritmos de optimización para configuraciones atómicas.

Este módulo implementa diferentes estrategias de optimización:
    - Búsqueda exhaustiva (fuerza bruta)
    - Simulated Annealing (MCMC con temperatura decreciente)
    - Esquemas de enfriamiento

Exports principales:
    - brute_force_search: Búsqueda exhaustiva
    - simulated_annealing: Algoritmo principal de SA
    - run_multiple_sa: Ejecuciones múltiples de SA
    - Esquemas de enfriamiento: geometric_cooling, exponential_cooling, etc.
"""

from .brute_force import (
    brute_force_search,
    get_top_k_positions,
    analyze_energy_landscape
)

from .simulated_annealing import (
    simulated_annealing,
    run_multiple_sa
)

from .cooling_schedules import (
    geometric_cooling,
    exponential_cooling,
    logarithmic_cooling,
    adaptive_cooling,
    linear_cooling,
    get_cooling_schedule
)

__all__ = [
    # Brute force
    'brute_force_search',
    'get_top_k_positions',
    'analyze_energy_landscape',

    # Simulated Annealing
    'simulated_annealing',
    'run_multiple_sa',

    # Cooling schedules
    'geometric_cooling',
    'exponential_cooling',
    'logarithmic_cooling',
    'adaptive_cooling',
    'linear_cooling',
    'get_cooling_schedule'
]
