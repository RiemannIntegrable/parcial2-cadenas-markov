"""
Submódulo para el Punto 1: Grilla 4x4 con 1 átomo de Ti.

Este submódulo implementa herramientas para resolver el problema de optimización
del Punto 1 del parcial, que consiste en encontrar la posición óptima de un
átomo de Titanio en una grilla 4x4 con 4 átomos de Nd en el centro.

Métodos utilizados:
    - Búsqueda exhaustiva (fuerza bruta): 12 configuraciones posibles
    - Simulated Annealing con diferentes esquemas de enfriamiento

Módulos:
    - morse: Potencial de Morse y parámetros atómicos
    - grid: Grillas 2D y cálculo de energía
    - optimization: Algoritmos de optimización
    - visualization: Visualización de resultados

Ejemplos de Uso:
    >>> from src.punto1 import create_grid_4x4, brute_force_search, simulated_annealing
    >>> from src.punto1 import plot_grid_configuration, plot_energy_evolution

    >>> # Crear grilla 4x4
    >>> grid = create_grid_4x4()

    >>> # Búsqueda exhaustiva
    >>> result_bf = brute_force_search(grid)
    >>> print(f"Óptimo: posición {result_bf['best_position_idx']}, "
    ...       f"energía {result_bf['best_energy']:.4f}")

    >>> # Simulated Annealing
    >>> from src.punto1.optimization import geometric_cooling
    >>> cooling = geometric_cooling(T0=2.0, alpha=0.95)
    >>> result_sa = simulated_annealing(grid, T0=2.0, cooling_schedule=cooling,
    ...                                 max_iter=5000, seed=42)

    >>> # Visualizar
    >>> grid.set_Ti_position(result_sa['best_position_idx'])
    >>> plot_grid_configuration(grid, title="Configuración Óptima")
    >>> plot_energy_evolution(result_sa['energy_history'],
    ...                       result_sa['temperature_history'],
    ...                       optimal_energy=result_bf['best_energy'])
"""

# Morse potential
from .morse import (
    morse_potential,
    morse_potential_from_types,
    get_morse_params,
    MORSE_PARAMETERS,
    ATOM_TYPES
)

# Grid
from .grid import (
    Grid2D,
    create_grid_4x4,
    get_Fe_positions_with_coords,
    compute_total_energy,
    compute_pairwise_energies,
    compute_Ti_contributions
)

# Optimization
from .optimization import (
    brute_force_search,
    get_top_k_positions,
    analyze_energy_landscape,
    simulated_annealing,
    run_multiple_sa,
    geometric_cooling,
    exponential_cooling,
    logarithmic_cooling,
    adaptive_cooling,
    linear_cooling,
    get_cooling_schedule
)

# Visualization
from .visualization import (
    plot_grid_configuration,
    plot_energy_evolution,
    plot_energy_distribution,
    plot_comparison_bar,
    plot_acceptance_rate,
    plot_multiple_runs
)

__version__ = '1.0.0'

__all__ = [
    # Morse
    'morse_potential',
    'morse_potential_from_types',
    'get_morse_params',
    'MORSE_PARAMETERS',
    'ATOM_TYPES',

    # Grid
    'Grid2D',
    'create_grid_4x4',
    'get_Fe_positions_with_coords',
    'compute_total_energy',
    'compute_pairwise_energies',
    'compute_Ti_contributions',

    # Optimization
    'brute_force_search',
    'get_top_k_positions',
    'analyze_energy_landscape',
    'simulated_annealing',
    'run_multiple_sa',
    'geometric_cooling',
    'exponential_cooling',
    'logarithmic_cooling',
    'adaptive_cooling',
    'linear_cooling',
    'get_cooling_schedule',

    # Visualization
    'plot_grid_configuration',
    'plot_energy_evolution',
    'plot_energy_distribution',
    'plot_comparison_bar',
    'plot_acceptance_rate',
    'plot_multiple_runs'
]
