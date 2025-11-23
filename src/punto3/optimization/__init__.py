"""
Módulo optimization: Simulated Annealing para optimización 3D

Provee funciones para:
- Algoritmo de Simulated Annealing con enfriamiento logarítmico
- Ejecución paralela de múltiples runs independientes
- Esquemas de enfriamiento (logarítmico, geométrico, etc.)
"""

from .sa_numba import (
    simulated_annealing_logarithmic_3d,
    simulated_annealing_core_logarithmic_3d
)

from .parallel_runs import (
    run_sa_single_logarithmic_3d,
    ejecutar_multiples_runs_logarithmic_3d,
    get_best_run,
    get_run_statistics
)

from .cooling_schedules import (
    logarithmic_cooling,
    geometric_cooling
)

__all__ = [
    'simulated_annealing_logarithmic_3d',
    'simulated_annealing_core_logarithmic_3d',
    'run_sa_single_logarithmic_3d',
    'ejecutar_multiples_runs_logarithmic_3d',
    'get_best_run',
    'get_run_statistics',
    'logarithmic_cooling',
    'geometric_cooling'
]
