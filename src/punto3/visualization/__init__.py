"""
Módulo visualization: Gráficas 3D interactivas con Plotly

Provee funciones para visualizar la estructura cristalina 3D y resultados de optimización.
"""

from .plotting_3d import (
    plot_crystal_configuration_3d,
    plot_energy_evolution_3d,
    plot_spatial_metrics_3d
)

__all__ = [
    'plot_crystal_configuration_3d',
    'plot_energy_evolution_3d',
    'plot_spatial_metrics_3d'
]
