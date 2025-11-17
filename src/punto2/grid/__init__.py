"""
Módulo grid - Representación y manipulación de la grilla 10×10.

Este módulo maneja la grilla del sistema usando arrays numpy puros
para compatibilidad con Numba y máximo rendimiento.
"""

from .grid_utils import (
    crear_grid_inicial,
    get_Fe_positions,
    get_Ti_positions,
    get_Nd_positions_fijas,
    aplicar_swap
)

from .energy_numba import (
    compute_total_energy_fast,
    compute_delta_E_swap_fast
)

__all__ = [
    'crear_grid_inicial',
    'get_Fe_positions',
    'get_Ti_positions',
    'get_Nd_positions_fijas',
    'aplicar_swap',
    'compute_total_energy_fast',
    'compute_delta_E_swap_fast'
]
