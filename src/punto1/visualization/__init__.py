"""
Módulo visualization: Visualización de resultados y análisis.

Este módulo proporciona funciones para visualizar:
    - Configuraciones de grillas con átomos
    - Evolución de energía en Simulated Annealing
    - Distribuciones de energía
    - Comparaciones entre métodos
    - Múltiples ejecuciones

Exports principales:
    - plot_grid_configuration: Visualizar grilla 2D
    - plot_energy_evolution: Evolución de SA
    - plot_energy_distribution: Histograma de energías
    - plot_comparison_bar: Comparar métodos
    - plot_acceptance_rate: Tasa de aceptación
    - plot_multiple_runs: Múltiples ejecuciones
"""

from .plotting import (
    plot_grid_configuration,
    plot_energy_evolution,
    plot_energy_distribution,
    plot_comparison_bar,
    plot_acceptance_rate,
    plot_multiple_runs,
    ATOM_COLORS,
    ATOM_SIZES
)

__all__ = [
    'plot_grid_configuration',
    'plot_energy_evolution',
    'plot_energy_distribution',
    'plot_comparison_bar',
    'plot_acceptance_rate',
    'plot_multiple_runs',
    'ATOM_COLORS',
    'ATOM_SIZES'
]
