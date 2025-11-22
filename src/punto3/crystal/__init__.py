"""Utilidades para manejar estructura cristalina 3D."""

from .crystal_utils import (
    get_Nd_positions_fijas,
    get_Fe_positions_all,
    crear_configuracion_inicial,
    get_Ti_indices_from_types
)

from .energy_numba import (
    compute_total_energy_fast_3d,
    compute_delta_E_swap_fast_3d,
    validate_delta_E_3d
)

__all__ = [
    'get_Nd_positions_fijas',
    'get_Fe_positions_all',
    'crear_configuracion_inicial',
    'get_Ti_indices_from_types',
    'compute_total_energy_fast_3d',
    'compute_delta_E_swap_fast_3d',
    'validate_delta_E_3d'
]
