"""
Módulo crystal: Gestión de la estructura cristalina 3D

Provee funciones para:
- Cargar coordenadas de Nd y Fe desde datos cristalográficos
- Crear configuraciones iniciales con Ti aleatorios
- Calcular energía total del sistema 3D
"""

from .crystal_utils import (
    get_Nd_positions_fixed,
    get_Fe_positions_all,
    crear_configuracion_inicial_3d,
    get_Ti_indices_from_types
)

from .energy_numba import (
    compute_total_energy_fast_3d,
    compute_energy_contribution_3d
)

__all__ = [
    'get_Nd_positions_fixed',
    'get_Fe_positions_all',
    'crear_configuracion_inicial_3d',
    'get_Ti_indices_from_types',
    'compute_total_energy_fast_3d',
    'compute_energy_contribution_3d'
]
