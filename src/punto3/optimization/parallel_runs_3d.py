"""
Ejecución paralela de múltiples runs de Simulated Annealing 3D usando joblib.

Este módulo permite ejecutar múltiples runs independientes del algoritmo
en paralelo, aprovechando todos los cores del CPU disponibles.

Speedup esperado: ~N× donde N = número de cores.
"""

import numpy as np
from joblib import Parallel, delayed
from typing import List, Dict, Optional
import warnings

# Suprimir warnings de Numba en paralelo
warnings.filterwarnings('ignore', category=Warning)


def run_sa_single_3d(
    run_id: int,
    Fe_candidate_positions: np.ndarray,
    Nd_positions: np.ndarray,
    c: float,
    t0: int,
    max_iterations: int,
    morse_params_array: np.ndarray,
    verbose: bool = False
) -> Dict:
    """
    Ejecuta un run independiente de Simulated Annealing 3D.

    Esta función es llamada por cada worker de joblib en paralelo.

    Args:
        run_id: ID del run (usado como semilla)
        Fe_candidate_positions: Array (96, 3) con posiciones candidatas de Fe
        Nd_positions: Array (16, 3) con posiciones de Nd
        c: Constante de enfriamiento logarítmico
        t0: Offset temporal
        max_iterations: Número de iteraciones
        morse_params_array: Parámetros de Morse
        verbose: Si True, imprime progreso

    Returns:
        Dict con resultados del run:
            - 'run_id': ID del run
            - 'atom_types_best': Mejor configuración encontrada
            - 'Ti_indices_best': Índices de Ti en mejor configuración
            - 'energia_final': Energía mínima alcanzada
            - 'energia_inicial': Energía inicial
            - 'mejora_relativa': (E_inicial - E_final) / |E_inicial|
            - 'energy_history': Historia de energías
            - 'iterations_to_best': Iteración donde se encontró el mejor
            - 'acceptance_rate': Tasa de aceptación
    """
    # Importar aquí para evitar problemas de imports en workers paralelos
    # Usar imports absolutos en lugar de relativos para compatibilidad con joblib
    import sys
    from pathlib import Path

    # Asegurar que el path está en sys.path
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from grid.config_utils_3d import crear_configuracion_inicial_3d
    from optimization.sa_numba_3d import simulated_annealing_logarithmic_3d

    if verbose:
        print(f"[Run {run_id}] Iniciando...")

    # Crear configuración inicial con semilla única para cada run
    atom_types_inicial, all_positions, Ti_indices_inicial, Fe_indices_inicial = \
        crear_configuracion_inicial_3d(Fe_candidate_positions, Nd_positions, seed=run_id)

    # Ejecutar Simulated Annealing
    resultado = simulated_annealing_logarithmic_3d(
        atom_types_inicial,
        all_positions,
        Ti_indices_inicial,
        Fe_indices_inicial,
        morse_params_array,
        c=c,
        t0=t0,
        max_iterations=max_iterations,
        seed=run_id,
        verbose=False  # Silencioso en paralelo
    )

    # Extraer métricas
    energia_inicial = resultado['energy_initial']
    energia_final = resultado['energy_best']
    mejora_relativa = resultado['energy_improvement_pct'] / 100

    # Iteración donde se encontró el mejor
    energy_history = resultado['energy_history']
    iterations_to_best = int(np.argmin(energy_history))

    if verbose:
        print(f"[Run {run_id}] Completado: E_inicial={energia_inicial:.4f}, "
              f"E_final={energia_final:.4f}, mejora={mejora_relativa*100:.2f}%")

    # Retornar resultados (incluir solo lo esencial para ahorrar memoria)
    return {
        'run_id': run_id,
        'atom_types_best': resultado['atom_types_best'],
        'Ti_indices_best': resultado['Ti_indices_best'],
        'energia_final': energia_final,
        'energia_inicial': energia_inicial,
        'mejora_relativa': mejora_relativa,
        'iterations_to_best': iterations_to_best,
        'acceptance_rate': resultado['acceptance_rate'],
        'energy_history': energy_history,  # Comentar si se quiere ahorrar memoria
        'elapsed_time': resultado['elapsed_time']
    }


def ejecutar_multiples_runs_3d(
    n_runs: int,
    Fe_candidate_positions: np.ndarray,
    Nd_positions: np.ndarray,
    c: float,
    t0: int,
    max_iterations: int,
    morse_params_array: np.ndarray,
    n_jobs: int = -1,
    verbose: int = 10
) -> List[Dict]:
    """
    Ejecuta múltiples runs de Simulated Annealing 3D en paralelo.

    Usa joblib para paralelizar la ejecución en múltiples cores.

    Args:
        n_runs: Número de runs independientes a ejecutar
        Fe_candidate_positions: Array (96, 3) con posiciones candidatas de Fe
        Nd_positions: Array (16, 3) con posiciones de Nd
        c: Constante de enfriamiento logarítmico
        t0: Offset temporal
        max_iterations: Número de iteraciones por run
        morse_params_array: Array (3, 3, 3) con parámetros de Morse
        n_jobs: Número de jobs paralelos
                -1: usa todos los cores disponibles
                1: ejecución secuencial (útil para debugging)
                N: usa N cores
        verbose: Nivel de verbosidad de joblib
                 0: silencioso
                 10: imprime progreso de cada job
                 50: muy verbose

    Returns:
        Lista de diccionarios con resultados de cada run

    Examples:
        >>> from src.punto3.morse import preparar_morse_params_array
        >>> params = preparar_morse_params_array()
        >>> resultados = ejecutar_multiples_runs_3d(
        ...     n_runs=64,
        ...     Fe_candidate_positions=Fe_pos,
        ...     Nd_positions=Nd_pos,
        ...     c=100.0,
        ...     t0=10,
        ...     max_iterations=1_000_000,
        ...     morse_params_array=params,
        ...     n_jobs=-1  # Usa todos los cores
        ... )
        >>> len(resultados)
        64
        >>> mejor = min(resultados, key=lambda x: x['energia_final'])
        >>> mejor['energia_final']
        -456.789...

    Note:
        - Cada run usa un seed diferente (run_id) para independencia
        - Los runs son completamente independientes y pueden ejecutarse en paralelo
        - El speedup esperado es ~N× donde N es el número de cores
        - En una máquina de 8 cores, 64 runs tardan ~8× el tiempo de 1 run
    """
    print(f"Ejecutando {n_runs} runs de Simulated Annealing 3D en paralelo...")
    print(f"  Parámetros: c={c}, t₀={t0}, max_iter={max_iterations:,}")
    print(f"  Jobs paralelos: {n_jobs} ({'todos los cores' if n_jobs == -1 else f'{n_jobs} cores'})")
    print()

    # Ejecutar en paralelo usando joblib
    resultados = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_sa_single_3d)(
            run_id=i,
            Fe_candidate_positions=Fe_candidate_positions,
            Nd_positions=Nd_positions,
            c=c,
            t0=t0,
            max_iterations=max_iterations,
            morse_params_array=morse_params_array,
            verbose=False  # Desactivar verbose individual (joblib ya muestra progreso)
        )
        for i in range(n_runs)
    )

    print(f"\n✓ Completados {len(resultados)} runs")

    # Estadísticas resumidas
    energias_finales = [r['energia_final'] for r in resultados]
    mejoras = [r['mejora_relativa'] for r in resultados]
    tiempos = [r['elapsed_time'] for r in resultados]

    print("\nEstadísticas:")
    print(f"  Mejor energía: {min(energias_finales):.6f}")
    print(f"  Peor energía: {max(energias_finales):.6f}")
    print(f"  Media: {np.mean(energias_finales):.6f}")
    print(f"  Desv. estándar: {np.std(energias_finales):.6f}")
    print(f"  Mejora promedio: {np.mean(mejoras)*100:.2f}%")
    print(f"  Tiempo promedio por run: {np.mean(tiempos):.2f} s ({np.mean(tiempos)/60:.2f} min)")

    return resultados


def get_best_run(resultados: List[Dict]) -> Dict:
    """
    Obtiene el mejor run de una lista de resultados.

    Args:
        resultados: Lista de diccionarios con resultados de runs

    Returns:
        Diccionario con el mejor run (menor energía final)

    Examples:
        >>> mejor = get_best_run(resultados)
        >>> mejor['run_id']
        42
        >>> mejor['energia_final']
        -456.789...
    """
    return min(resultados, key=lambda x: x['energia_final'])


def get_run_statistics(resultados: List[Dict]) -> Dict:
    """
    Calcula estadísticas sobre múltiples runs.

    Args:
        resultados: Lista de diccionarios con resultados de runs

    Returns:
        Dict con estadísticas:
            - 'n_runs': Número de runs
            - 'energia_min': Mejor energía encontrada
            - 'energia_max': Peor energía encontrada
            - 'energia_mean': Media de energías
            - 'energia_std': Desviación estándar
            - 'energia_median': Mediana de energías
            - 'mejora_mean': Mejora promedio (%)
            - 'mejora_std': Desviación estándar de mejora
            - 'acceptance_rate_mean': Tasa de aceptación promedio
    """
    energias_finales = np.array([r['energia_final'] for r in resultados])
    mejoras = np.array([r['mejora_relativa'] * 100 for r in resultados])
    acceptance_rates = np.array([r['acceptance_rate'] for r in resultados])

    return {
        'n_runs': len(resultados),
        'energia_min': np.min(energias_finales),
        'energia_max': np.max(energias_finales),
        'energia_mean': np.mean(energias_finales),
        'energia_std': np.std(energias_finales),
        'energia_median': np.median(energias_finales),
        'mejora_mean': np.mean(mejoras),
        'mejora_std': np.std(mejoras),
        'acceptance_rate_mean': np.mean(acceptance_rates),
        'acceptance_rate_std': np.std(acceptance_rates)
    }


def filter_top_runs(resultados: List[Dict], top_n: int = 10) -> List[Dict]:
    """
    Filtra los mejores N runs por energía final.

    Args:
        resultados: Lista de diccionarios con resultados de runs
        top_n: Número de mejores runs a retornar

    Returns:
        Lista con los top_n mejores runs ordenados por energía (menor a mayor)

    Examples:
        >>> top_10 = filter_top_runs(resultados, top_n=10)
        >>> len(top_10)
        10
        >>> top_10[0]['energia_final'] < top_10[1]['energia_final']
        True
    """
    sorted_results = sorted(resultados, key=lambda x: x['energia_final'])
    return sorted_results[:top_n]
