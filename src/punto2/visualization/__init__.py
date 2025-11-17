"""
Módulo visualization - Gráficas para el Problema 2.

Contiene funciones para crear las visualizaciones requeridas en el Punto 3:
- Configuración óptima de la grilla 10×10
- Evolución de energía vs iteración
- Identificación de fases de exploración/explotación
- Comparación de múltiples runs
"""

from .plotting import (
    plot_grid_configuration,
    plot_energy_evolution,
    plot_multiple_runs_comparison,
    plot_spatial_metrics,
    plot_acceptance_rate
)

__all__ = [
    'plot_grid_configuration',
    'plot_energy_evolution',
    'plot_multiple_runs_comparison',
    'plot_spatial_metrics',
    'plot_acceptance_rate'
]
