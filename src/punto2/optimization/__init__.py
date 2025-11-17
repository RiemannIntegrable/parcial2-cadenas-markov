"""
Módulo optimization - Algoritmos de optimización para el Problema 2.

Contiene implementaciones de Simulated Annealing optimizadas con Numba
y utilidades para ejecución paralela.
"""

from .sa_numba import simulated_annealing, simulated_annealing_core
from .cooling_schedules import (
    geometric_cooling,
    exponential_cooling,
    logarithmic_cooling,
    linear_cooling,
    adaptive_cooling
)

__all__ = [
    'simulated_annealing',
    'simulated_annealing_core',
    'geometric_cooling',
    'exponential_cooling',
    'logarithmic_cooling',
    'linear_cooling',
    'adaptive_cooling'
]
