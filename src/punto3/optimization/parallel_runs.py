"""
Ejecución paralela de múltiples runs de Simulated Annealing para estructura 3D usando joblib.

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


def run_sa_single_logarithmic_3d(
    run_id: int,
    c: float,
    t0: int,
    max_iterations: int,
    morse_params_array: np.ndarray,
    all_positions: np.ndarray,
    save_every: int = 10,
    verbose: bool = False
) -> Dict:
    """
    Ejecuta un run independiente de Simulated Annealing logarítmico en 3D.

    Esta función es llamada por cada worker de joblib en paralelo.

    Args:
        run_id: ID del run (usado como semilla)
        c: Constante de enfriamiento logarítmico
        t0: Offset temporal
        max_iterations: Número de iteraciones
        morse_params_array: Parámetros de Morse
        all_positions: Array (112, 3) con coordenadas (compartido entre runs)
        save_every: Guardar historia cada N iteraciones
        verbose: Si True, imprime progreso

    Returns:
        Dict con resultados del run:
            - 'run_id': ID del run
            - 'atom_types_best': Mejor configuración encontrada
            - 'Ti_indices_best': Índices de Ti en mejor configuración
            - 'energia_final': Energía mínima alcanzada
            - 'energia_inicial': Energía inicial
            - 'mejora_relativa': (E_inicial - E_final) / E_inicial
            - 'energy_history': Historia submuestreada de energías
            - 'iterations_to_best': Iteración donde se encontró el mejor
    """
    # Importar aquí para evitar problemas de imports en workers paralelos
    from ..crystal.crystal_utils import crear_configuracion_inicial
    from ..crystal.energy_numba import compute_total_energy_fast_3d
    from .sa_numba import simulated_annealing_logarithmic_3d

    if verbose:
        print(f"[Run {run_id}] Iniciando...")

    # Crear configuración inicial con semilla única para cada run
    all_pos, atom_types_init, Ti_indices_init, Fe_indices_init, _ = crear_configuracion_inicial(seed=run_id)

    # Calcular energía inicial (para métricas)
    energia_inicial = compute_total_energy_fast_3d(all_pos, atom_types_init, morse_params_array)

    # Ejecutar Simulated Annealing
    atom_types_best, Ti_indices_best, history = simulated_annealing_logarithmic_3d(
        all_pos,
        atom_types_init,
        Ti_indices_init,
        Fe_indices_init,
        morse_params_array,
        c=c,
        t0=t0,
        max_iterations=max_iterations,
        seed=run_id,
        save_every=save_every
    )

    # Energía final
    energia_final = history['energy_best']

    # Iteración donde se encontró el mejor
    iterations_to_best = history['iterations_to_best']

    # Mejora relativa
    mejora_relativa = (energia_inicial - energia_final) / abs(energia_inicial)

    if verbose:
        print(f"[Run {run_id}] Completado: E_inicial={energia_inicial:.4f}, "
              f"E_final={energia_final:.4f}, mejora={mejora_relativa*100:.2f}%")

    # Retornar resultados (minimizar memoria: no guardar history completo)
    return {
        'run_id': run_id,
        'atom_types_best': atom_types_best,
        'Ti_indices_best': Ti_indices_best,
        'energia_final': energia_final,
        'energia_inicial': energia_inicial,
        'mejora_relativa': mejora_relativa,
        'iterations_to_best': iterations_to_best,
        'energy_history': history['energy']  # Submuestreado
    }


def ejecutar_multiples_runs_logarithmic_3d(
    n_runs: int,
    c: float,
    t0: int,
    max_iterations: int,
    morse_params_array: np.ndarray,
    save_every: int = 10,
    n_jobs: int = -1,
    verbose: int = 10
) -> List[Dict]:
    """
    Ejecuta múltiples runs de Simulated Annealing logarítmico en 3D en paralelo.

    Usa joblib para paralelizar la ejecución en múltiples cores.

    Args:
        n_runs: Número de runs independientes a ejecutar
        c: Constante de enfriamiento logarítmico
        t0: Offset temporal
        max_iterations: Número de iteraciones por run
        morse_params_array: Array (3, 3, 3) con parámetros de Morse
        save_every: Guardar historia cada N iteraciones (default 10)
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
        >>> resultados = ejecutar_multiples_runs_logarithmic_3d(
        ...     n_runs=10,
        ...     c=3000,
        ...     t0=2,
        ...     max_iterations=1000000,
        ...     morse_params_array=params,
        ...     n_jobs=-1  # Usa todos los cores
        ... )
        >>> len(resultados)
        10
        >>> mejor = min(resultados, key=lambda x: x['energia_final'])
    """
    # Obtener posiciones (compartidas entre todos los runs)
    from ..crystal.crystal_utils import get_Nd_positions_fijas, get_Fe_positions_all

    Nd_pos = get_Nd_positions_fijas()
    Fe_pos = get_Fe_positions_all()
    all_positions = np.vstack([Nd_pos, Fe_pos]).astype(np.float32)

    print(f"Ejecutando {n_runs} runs de Simulated Annealing (logarítmico 3D) en paralelo...")
    print(f"  Parámetros: c={c}, t₀={t0}, max_iter={max_iterations:,}")
    print(f"  Jobs paralelos: {n_jobs} ({'todos los cores' if n_jobs == -1 else f'{n_jobs} cores'})")
    print()

    # Ejecutar runs en paralelo usando joblib
    resultados = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_sa_single_logarithmic_3d)(
            run_id=i,
            c=c,
            t0=t0,
            max_iterations=max_iterations,
            morse_params_array=morse_params_array,
            all_positions=all_positions,
            save_every=save_every,
            verbose=False  # Desactivar prints individuales para no saturar salida
        )
        for i in range(n_runs)
    )

    print(f"\n✓ Completados {n_runs} runs")

    # Estadísticas rápidas
    energias = [r['energia_final'] for r in resultados]
    mejoras = [r['mejora_relativa'] for r in resultados]

    print(f"\nEstadísticas:")
    print(f"  Mejor energía: {np.min(energias):.6f}")
    print(f"  Peor energía: {np.max(energias):.6f}")
    print(f"  Media: {np.mean(energias):.6f}")
    print(f"  Desv. estándar: {np.std(energias):.6f}")
    print(f"  Mejora promedio: {np.mean(mejoras)*100:.2f}%")

    return resultados


def get_best_run(resultados: List[Dict]) -> Dict:
    """
    Obtiene el run con la mejor energía final.

    Args:
        resultados: Lista de diccionarios de resultados

    Returns:
        Diccionario del mejor run

    Examples:
        >>> mejor = get_best_run(resultados)
        >>> mejor['energia_final']
        48679.97...
    """
    return min(resultados, key=lambda x: x['energia_final'])


def get_run_statistics(resultados: List[Dict]) -> Dict:
    """
    Calcula estadísticas agregadas de múltiples runs.

    Args:
        resultados: Lista de diccionarios de resultados

    Returns:
        Diccionario con estadísticas:
            - 'energia_media': Media de energías finales
            - 'energia_std': Desviación estándar
            - 'energia_min': Mejor energía encontrada
            - 'energia_max': Peor energía encontrada
            - 'mejora_relativa_media': Mejora promedio
            - 'convergencia_media': Iteración promedio de convergencia

    Examples:
        >>> stats = get_run_statistics(resultados)
        >>> stats['energia_min']
        48679.97...
    """
    energias = np.array([r['energia_final'] for r in resultados])
    mejoras = np.array([r['mejora_relativa'] for r in resultados])
    convergencias = np.array([r['iterations_to_best'] for r in resultados])

    return {
        'energia_media': float(np.mean(energias)),
        'energia_std': float(np.std(energias)),
        'energia_min': float(np.min(energias)),
        'energia_max': float(np.max(energias)),
        'mejora_relativa_media': float(np.mean(mejoras)),
        'convergencia_media': float(np.mean(convergencias)),
        'convergencia_mediana': float(np.median(convergencias))
    }
