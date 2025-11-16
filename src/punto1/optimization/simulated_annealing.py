"""
Implementación de Simulated Annealing para optimización de configuraciones.

Este módulo implementa el algoritmo de Recocido Simulado basado en el
criterio de Metropolis-Hastings, como se describe en la clase teórica.
"""

import numpy as np
from typing import Callable, Dict, Optional
from ..grid import Grid2D, compute_total_energy


def simulated_annealing(
    grid: Grid2D,
    T0: float = 1.0,
    cooling_schedule: Optional[Callable[[int], float]] = None,
    max_iter: int = 10000,
    initial_position: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = False
) -> Dict:
    """
    Algoritmo de Simulated Annealing para encontrar la posición óptima del Ti.

    Implementa el algoritmo de Metropolis-Hastings con temperatura decreciente
    según el esquema de enfriamiento especificado.

    Args:
        grid: Grilla 2D base (sin Ti asignado, o se reasignará)
        T0: Temperatura inicial
        cooling_schedule: Función iteración → temperatura.
                         Si None, usa geométrico con α=0.95
        max_iter: Número máximo de iteraciones
        initial_position: Posición inicial del Ti (None = aleatoria)
        seed: Semilla para reproducibilidad
        verbose: Si imprimir progreso cada 1000 iteraciones

    Returns:
        Diccionario con:
            - 'best_position_idx': Mejor posición encontrada
            - 'best_energy': Mejor energía encontrada
            - 'final_position_idx': Posición al final de la ejecución
            - 'final_energy': Energía final
            - 'energy_history': Lista de energías en cada iteración
            - 'temperature_history': Lista de temperaturas
            - 'acceptance_rate': Tasa de aceptación global
            - 'improvement_rate': Tasa de movimientos que mejoraron
            - 'n_iterations': Número de iteraciones ejecutadas

    Examples:
        >>> from src.grid import create_grid_4x4
        >>> from src.optimization.cooling_schedules import geometric_cooling
        >>> grid = create_grid_4x4()
        >>> cooling = geometric_cooling(T0=2.0, alpha=0.95)
        >>> result = simulated_annealing(grid, T0=2.0, cooling_schedule=cooling,
        ...                              max_iter=5000, seed=42)
        >>> result['n_iterations']
        5000
        >>> 0 <= result['best_position_idx'] < 12
        True

    Note:
        El algoritmo implementa:
        1. Propuesta: elegir nueva posición uniformemente al azar
        2. Aceptación: criterio de Metropolis
           α = min(1, exp(-ΔE/T))
        3. Enfriamiento: actualizar T según schedule
    """
    # Configurar RNG
    rng = np.random.default_rng(seed)

    # Crear copia de trabajo
    working_grid = grid.copy()

    # Esquema de enfriamiento por defecto: geométrico
    if cooling_schedule is None:
        alpha = 0.95
        cooling_schedule = lambda t: T0 * (alpha ** t)

    # Número de posiciones candidatas
    n_positions = working_grid.n_Fe_sites

    # Estado inicial
    if initial_position is None:
        current_position = rng.integers(0, n_positions)
    else:
        current_position = initial_position

    working_grid.set_Ti_position(current_position)
    current_energy = compute_total_energy(working_grid)

    # Mejor estado encontrado
    best_position = current_position
    best_energy = current_energy

    # Historias para análisis
    energy_history = [current_energy]
    temperature_history = [T0]

    # Contadores
    n_accepted = 0
    n_improved = 0

    # Loop principal de SA
    for iteration in range(1, max_iter + 1):

        # Proponer nueva posición (uniformemente al azar)
        proposed_position = rng.integers(0, n_positions)

        # Calcular energía de la propuesta
        working_grid.set_Ti_position(proposed_position)
        proposed_energy = compute_total_energy(working_grid)

        # Cambio de energía
        delta_E = proposed_energy - current_energy

        # Temperatura actual
        T = cooling_schedule(iteration)

        # Criterio de Metropolis-Hastings
        accept = False

        if delta_E <= 0:
            # Siempre aceptar mejoras
            accept = True
            n_improved += 1
        else:
            # Aceptar empeoramientos probabilísticamente
            acceptance_probability = np.exp(-delta_E / T)
            if rng.random() < acceptance_probability:
                accept = True

        # Aplicar movimiento si se acepta
        if accept:
            current_position = proposed_position
            current_energy = proposed_energy
            n_accepted += 1

            # Actualizar mejor solución si es necesario
            if current_energy < best_energy:
                best_position = current_position
                best_energy = current_energy
        else:
            # Rechazar: volver a posición anterior
            working_grid.set_Ti_position(current_position)

        # Registro
        energy_history.append(current_energy)
        temperature_history.append(T)

        # Imprimir progreso
        if verbose and iteration % 1000 == 0:
            acceptance_rate = n_accepted / iteration
            improvement_rate = n_improved / iteration
            print(
                f"Iter {iteration:6d} | T={T:.6f} | "
                f"E_current={current_energy:.4f} | E_best={best_energy:.4f} | "
                f"Accept={acceptance_rate:.3f} | Improve={improvement_rate:.3f}"
            )

    # Calcular tasas finales
    total_acceptance_rate = n_accepted / max_iter
    total_improvement_rate = n_improved / max_iter

    return {
        'best_position_idx': best_position,
        'best_energy': best_energy,
        'final_position_idx': current_position,
        'final_energy': current_energy,
        'energy_history': energy_history,
        'temperature_history': temperature_history,
        'acceptance_rate': total_acceptance_rate,
        'improvement_rate': total_improvement_rate,
        'n_iterations': max_iter
    }


def run_multiple_sa(
    grid: Grid2D,
    n_runs: int = 10,
    **sa_kwargs
) -> Dict:
    """
    Ejecuta Simulated Annealing múltiples veces y agrega resultados.

    Útil para evaluar robustez y variabilidad del algoritmo.

    Args:
        grid: Grilla 2D
        n_runs: Número de ejecuciones independientes
        **sa_kwargs: Argumentos para simulated_annealing()

    Returns:
        Diccionario con:
            - 'results': Lista de resultados de cada ejecución
            - 'best_overall_energy': Mejor energía entre todas las ejecuciones
            - 'best_overall_position': Posición correspondiente
            - 'mean_best_energy': Promedio de mejores energías
            - 'std_best_energy': Desviación estándar
            - 'success_rate': % de veces que encontró el mejor resultado

    Examples:
        >>> from src.grid import create_grid_4x4
        >>> grid = create_grid_4x4()
        >>> multi_result = run_multiple_sa(grid, n_runs=5, max_iter=1000, seed=42)
        >>> len(multi_result['results'])
        5
    """
    results = []
    best_energies = []

    for run_idx in range(n_runs):
        # Usar semilla diferente para cada run si se especificó una base
        if 'seed' in sa_kwargs and sa_kwargs['seed'] is not None:
            sa_kwargs['seed'] = sa_kwargs['seed'] + run_idx

        result = simulated_annealing(grid, **sa_kwargs)
        results.append(result)
        best_energies.append(result['best_energy'])

    best_energies = np.array(best_energies)

    # Encontrar el mejor global
    best_run_idx = np.argmin(best_energies)
    best_overall_energy = best_energies[best_run_idx]
    best_overall_position = results[best_run_idx]['best_position_idx']

    # Calcular success rate (qué tan seguido encontró el mejor)
    tolerance = 1e-6
    n_success = np.sum(np.abs(best_energies - best_overall_energy) < tolerance)
    success_rate = n_success / n_runs

    return {
        'results': results,
        'best_overall_energy': best_overall_energy,
        'best_overall_position': best_overall_position,
        'mean_best_energy': float(np.mean(best_energies)),
        'std_best_energy': float(np.std(best_energies)),
        'success_rate': success_rate
    }
