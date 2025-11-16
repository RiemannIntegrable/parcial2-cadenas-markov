"""
Módulo grid: Grillas 2D y cálculo de energía.

Este módulo maneja la creación y manipulación de grillas 2D para el
Problema 1, así como el cálculo de energías usando el potencial de Morse.

Exports principales:
    - Grid2D: Clase para representar grillas
    - create_grid_4x4: Crea la grilla del problema
    - compute_total_energy: Calcula energía total
    - compute_pairwise_energies: Análisis de pares
    - compute_Ti_contributions: Análisis de contribuciones del Ti
"""

from .grid_2d import (
    Grid2D,
    create_grid_4x4,
    get_Fe_positions_with_coords
)

from .energy import (
    compute_distance_2d,
    compute_total_energy,
    compute_pairwise_energies,
    compute_Ti_contributions
)

__all__ = [
    'Grid2D',
    'create_grid_4x4',
    'get_Fe_positions_with_coords',
    'compute_distance_2d',
    'compute_total_energy',
    'compute_pairwise_energies',
    'compute_Ti_contributions'
]
