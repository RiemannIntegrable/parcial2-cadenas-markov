"""
Ejecución paralela de múltiples runs de Simulated Annealing usando joblib.

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


def run_sa_single(
    run_id: int,
    T0: float,
    alpha: float,
    max_iterations: int,
    morse_params_array: np.ndarray,
    verbose: bool = False
) -> Dict:
    """
    Ejecuta un run independiente de Simulated Annealing.

    Esta función es llamada por cada worker de joblib en paralelo.

    Args:
        run_id: ID del run (usado como semilla)
        T0: Temperatura inicial
        alpha: Factor de enfriamiento
        max_iterations: Número de iteraciones
        morse_params_array: Parámetros de Morse
        verbose: Si True, imprime progreso

    Returns:
        Dict con resultados del run:
            - 'run_id': ID del run
            - 'grid_best': Mejor configuración encontrada
            - 'Ti_best': Posiciones de Ti en mejor configuración
            - 'energia_final': Energía mínima alcanzada
            - 'energia_inicial': Energía inicial
            - 'mejora_relativa': (E_inicial - E_final) / E_inicial
            - 'energy_history': Historia de energías (opcional, solo si se quiere)
            - 'iterations_to_best': Iteración donde se encontró el mejor
    """
    # Importar aquí para evitar problemas de imports en workers paralelos
    from ..grid.grid_utils import crear_grid_inicial
    from .sa_numba import simulated_annealing

    if verbose:
        print(f"[Run {run_id}] Iniciando...")

    # Crear grid inicial con semilla única para cada run
    grid_inicial, Ti_inicial, _ = crear_grid_inicial(seed=run_id)

    # Calcular energía inicial (para métricas)
    from ..grid.energy_numba import compute_total_energy_fast
    energia_inicial = compute_total_energy_fast(grid_inicial, morse_params_array)

    # Ejecutar Simulated Annealing
    grid_best, Ti_best, history = simulated_annealing(
        grid_inicial,
        Ti_inicial,
        morse_params_array,
        T0=T0,
        alpha=alpha,
        max_iterations=max_iterations,
        seed=run_id
    )

    # Energía final
    energia_final = history['energy_best']

    # Iteración donde se encontró el mejor
    iterations_to_best = int(np.argmin(history['energy']))

    # Mejora relativa
    mejora_relativa = (energia_inicial - energia_final) / abs(energia_inicial)

    if verbose:
        print(f"[Run {run_id}] Completado: E_inicial={energia_inicial:.4f}, "
              f"E_final={energia_final:.4f}, mejora={mejora_relativa*100:.2f}%")

    # Retornar resultados (minimizar memoria: no guardar history completo)
    return {
        'run_id': run_id,
        'grid_best': grid_best,
        'Ti_best': Ti_best,
        'energia_final': energia_final,
        'energia_inicial': energia_inicial,
        'mejora_relativa': mejora_relativa,
        'iterations_to_best': iterations_to_best,
        # NO guardar 'accepted_history' ni 'temperature_history' para ahorrar memoria
        # Solo guardar 'energy_history' si realmente se necesita
        'energy_history': history['energy']  # Comentar esta línea si se quiere ahorrar más memoria
    }


def ejecutar_multiples_runs(
    n_runs: int,
    T0: float,
    alpha: float,
    max_iterations: int,
    morse_params_array: np.ndarray,
    n_jobs: int = -1,
    verbose: int = 10
) -> List[Dict]:
    """
    Ejecuta múltiples runs de Simulated Annealing en paralelo.

    Usa joblib para paralelizar la ejecución en múltiples cores.

    Args:
        n_runs: Número de runs independientes a ejecutar
        T0: Temperatura inicial
        alpha: Factor de enfriamiento geométrico
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
        >>> from src.punto2.morse import preparar_morse_params_array
        >>> params = preparar_morse_params_array()
        >>> resultados = ejecutar_multiples_runs(
        ...     n_runs=10,
        ...     T0=20.0,
        ...     alpha=0.98,
        ...     max_iterations=50000,
        ...     morse_params_array=params,
        ...     n_jobs=-1  # Usa todos los cores
        ... )
        >>> len(resultados)
        10
        >>> mejor = min(resultados, key=lambda x: x['energia_final'])
        >>> mejor['energia_final']
        -123.456...

    Note:
        - Cada run usa un seed diferente (run_id) para independencia
        - Los runs son completamente independientes y pueden ejecutarse en paralelo
        - El speedup esperado es ~N× donde N es el número de cores
        - En una máquina de 8 cores, 10 runs tardan ~1.25× el tiempo de 1 run
    """
    print(f"Ejecutando {n_runs} runs de Simulated Annealing en paralelo...")
    print(f"  Parámetros: T0={T0}, α={alpha}, max_iter={max_iterations}")
    print(f"  Jobs paralelos: {n_jobs} ({'todos los cores' if n_jobs == -1 else f'{n_jobs} cores'})")
    print()

    # Ejecutar en paralelo usando joblib
    resultados = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_sa_single)(
            run_id=i,
            T0=T0,
            alpha=alpha,
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

    print("\nEstadísticas:")
    print(f"  Mejor energía: {min(energias_finales):.6f}")
    print(f"  Peor energía: {max(energias_finales):.6f}")
    print(f"  Media: {np.mean(energias_finales):.6f}")
    print(f"  Desv. estándar: {np.std(energias_finales):.6f}")
    print(f"  Mejora promedio: {np.mean(mejoras)*100:.2f}%")

    return resultados


def get_best_run(resultados: List[Dict]) -> Dict:
    """
    Obtiene el mejor run de una lista de resultados.

    Args:
        resultados: Lista de diccionarios con resultados de runs

    Returns:
        Diccionario del run con menor energía final

    Examples:
        >>> mejor = get_best_run(resultados)
        >>> mejor['run_id']
        7
        >>> mejor['energia_final']
        -123.456
    """
    return min(resultados, key=lambda x: x['energia_final'])


def get_run_statistics(resultados: List[Dict]) -> Dict[str, float]:
    """
    Calcula estadísticas sobre múltiples runs.

    Args:
        resultados: Lista de diccionarios con resultados de runs

    Returns:
        Dict con estadísticas:
            - 'energia_min': Mejor energía encontrada
            - 'energia_max': Peor energía encontrada
            - 'energia_media': Media de energías finales
            - 'energia_std': Desviación estándar
            - 'mejora_media': Mejora relativa promedio
            - 'convergencia_media': Iteración promedio para encontrar mejor

    Examples:
        >>> stats = get_run_statistics(resultados)
        >>> stats['energia_min']
        -123.456
        >>> stats['energia_media']
        -120.234
    """
    energias = [r['energia_final'] for r in resultados]
    mejoras = [r['mejora_relativa'] for r in resultados]
    convergencias = [r['iterations_to_best'] for r in resultados]

    return {
        'energia_min': float(np.min(energias)),
        'energia_max': float(np.max(energias)),
        'energia_media': float(np.mean(energias)),
        'energia_std': float(np.std(energias)),
        'mejora_media': float(np.mean(mejoras)),
        'convergencia_media': float(np.mean(convergencias)),
        'n_runs': len(resultados)
    }


def filter_top_runs(resultados: List[Dict], top_n: int = 5) -> List[Dict]:
    """
    Filtra los mejores N runs por energía final.

    Args:
        resultados: Lista de resultados
        top_n: Número de mejores runs a retornar

    Returns:
        Lista con los top_n mejores runs (menor energía)

    Examples:
        >>> top_5 = filter_top_runs(resultados, top_n=5)
        >>> len(top_5)
        5
        >>> top_5[0]['energia_final'] <= top_5[1]['energia_final']
        True
    """
    sorted_results = sorted(resultados, key=lambda x: x['energia_final'])
    return sorted_results[:top_n]
