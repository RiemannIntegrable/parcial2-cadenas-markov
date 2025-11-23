"""
Simulated Annealing optimizado con Numba para estructura cristalina 3D.

Este módulo implementa el algoritmo de Simulated Annealing adaptado para trabajar
con índices globales en lugar de posiciones de grilla.

DIFERENCIA CLAVE con punto2:
- Punto2: Swap de POSICIONES en grilla 2D (x,y)
- Punto3: Swap de ÍNDICES en atom_types (tipo de átomo en cada posición fija 3D)
"""

import numpy as np
from numba import njit
from typing import Dict, Tuple, Optional


@njit(fastmath=True)
def simulated_annealing_core_logarithmic_3d(
    all_positions: np.ndarray,      # (112, 3) - FIJO durante todo el SA
    atom_types: np.ndarray,         # (112,) - SE MODIFICA (swap de tipos)
    Ti_indices: np.ndarray,         # (8,) - SE ACTUALIZA
    morse_params_array: np.ndarray,  # (3, 3, 3)
    c: float,
    t0: int,
    max_iterations: int,
    seed: int
) -> tuple:
    """
    Core del algoritmo de Simulated Annealing con enfriamiento LOGARÍTMICO en 3D.

    Optimiza la configuración de 8 átomos de Ti intercambiándolos con átomos de Fe.
    Usa enfriamiento logarítmico: T(t) = c / log(t + t₀)

    Args:
        all_positions: Array (112, 3) con TODAS las posiciones atómicas (FIJAS)
        atom_types: Array (112,) con tipos de átomos [0=Fe, 1=Nd, 2=Ti]
        Ti_indices: Array (8,) con índices globales de los 8 átomos de Ti
        morse_params_array: Array (3, 3, 3) con parámetros de Morse
        c: Constante de enfriamiento (debe ser ≥ profundidad de barreras)
        t0: Offset temporal (típicamente 2)
        max_iterations: Número máximo de iteraciones
        seed: Semilla para reproducibilidad

    Returns:
        Tupla (atom_types_best, Ti_best, energy_history, accepted_history, temperature_history)
        - atom_types_best: Array (112,) con mejor configuración encontrada
        - Ti_best: Array (8,) con índices de Ti en mejor configuración
        - energy_history: Array (max_iterations,) con energías en cada iteración
        - accepted_history: Array (max_iterations,) bool con aceptaciones
        - temperature_history: Array (max_iterations,) con temperaturas

    Note:
        Las posiciones (all_positions) NO cambian. Solo cambian los TIPOS de átomos
        en cada posición fija. Esto es fundamentalmente diferente a punto2 donde
        las posiciones sí cambiaban en una grilla discreta.
    """
    # Inicializar random number generator
    np.random.seed(seed)

    # Copiar arrays para no modificar los originales
    atom_types_current = atom_types.copy()
    Ti_current = Ti_indices.copy()

    n_Ti = len(Ti_current)  # Debe ser 8

    # Calcular energía inicial
    E_current = compute_total_energy_3d(all_positions, atom_types_current, morse_params_array)

    # Mejor configuración encontrada
    atom_types_best = atom_types_current.copy()
    Ti_best = Ti_current.copy()
    E_best = E_current

    # Historia (pre-asignada para eficiencia)
    energy_history = np.zeros(max_iterations, dtype=np.float64)
    accepted_history = np.zeros(max_iterations, dtype=np.bool_)
    temperature_history = np.zeros(max_iterations, dtype=np.float64)

    # ========================================================================
    # LOOP PRINCIPAL DE SIMULATED ANNEALING
    # ========================================================================
    for iteration in range(max_iterations):
        # Calcular temperatura (enfriamiento LOGARÍTMICO)
        # T(t) = c / log(t + t₀)
        # Safety check: asegurar que el logaritmo no sea muy pequeño
        log_val = np.log(float(iteration + t0))
        if log_val < 0.1:
            log_val = 0.1
        T = c / log_val
        temperature_history[iteration] = T

        # ====================================================================
        # PROPONER MOVIMIENTO: Swap Ti ↔ Fe
        # ====================================================================

        # 1. Seleccionar Ti aleatorio por índice LOCAL (0..7)
        ti_local_idx = np.random.randint(0, n_Ti)
        ti_global_idx = Ti_current[ti_local_idx]

        # 2. Encontrar TODOS los Fe (candidatos para swap)
        # CRÍTICO: Usar np.where para obtener índices de Fe
        Fe_candidates = np.where(atom_types_current == 0)[0]
        n_Fe = len(Fe_candidates)

        if n_Fe == 0:
            # No hay Fe disponibles (error crítico)
            break

        # 3. Seleccionar Fe aleatorio
        fe_local_idx = np.random.randint(0, n_Fe)
        fe_global_idx = Fe_candidates[fe_local_idx]

        # ====================================================================
        # CALCULAR ΔE INCREMENTAL (OPTIMIZACIÓN CRÍTICA)
        # ====================================================================

        # Energía ANTES del swap (solo contribuciones de ti y fe)
        E_before_ti = compute_energy_contribution_3d(
            ti_global_idx, all_positions, atom_types_current, morse_params_array
        )
        E_before_fe = compute_energy_contribution_3d(
            fe_global_idx, all_positions, atom_types_current, morse_params_array
        )

        # HACER EL SWAP (temporalmente)
        atom_types_current[ti_global_idx] = 0  # Ti → Fe
        atom_types_current[fe_global_idx] = 2  # Fe → Ti

        # Energía DESPUÉS del swap
        E_after_ti = compute_energy_contribution_3d(
            ti_global_idx, all_positions, atom_types_current, morse_params_array
        )
        E_after_fe = compute_energy_contribution_3d(
            fe_global_idx, all_positions, atom_types_current, morse_params_array
        )

        # ΔE (incremental)
        delta_E = (E_after_ti + E_after_fe) - (E_before_ti + E_before_fe)

        # ====================================================================
        # CRITERIO DE METROPOLIS-HASTINGS
        # ====================================================================
        accept = False

        if delta_E < 0:
            # Mejora la energía → SIEMPRE aceptar
            accept = True
        elif T > 1e-10:
            # Aceptar con probabilidad exp(-ΔE/T)
            prob_aceptacion = np.exp(-delta_E / T)
            if np.random.random() < prob_aceptacion:
                accept = True

        # ====================================================================
        # APLICAR O REVERTIR MOVIMIENTO
        # ====================================================================
        if accept:
            # Mantener el swap
            # atom_types_current ya está actualizado arriba
            Ti_current[ti_local_idx] = fe_global_idx  # Actualizar índice de Ti
            E_current += delta_E

            # Actualizar mejor solución si es necesario
            if E_current < E_best:
                E_best = E_current
                atom_types_best[:] = atom_types_current
                Ti_best[:] = Ti_current

            accepted_history[iteration] = True
        else:
            # REVERTIR el swap
            atom_types_current[ti_global_idx] = 2  # Volver a Ti
            atom_types_current[fe_global_idx] = 0  # Volver a Fe
            # Ti_current no cambió, no hay que revertir

        # Guardar energía actual
        energy_history[iteration] = E_current

    # VERIFICACIÓN FINAL DE SANIDAD
    # CRÍTICO: Asegurar que no perdimos Ti
    n_Ti_final = 0
    for i in range(len(atom_types_best)):
        if atom_types_best[i] == 2:
            n_Ti_final += 1

    if n_Ti_final != n_Ti:
        # Este error NO debería ocurrir si la lógica es correcta
        # Pero es una salvaguarda crítica
        print(f"ERROR: Se perdieron Ti! Inicial: {n_Ti}, Final: {n_Ti_final}")

    return atom_types_best, Ti_best, energy_history, accepted_history, temperature_history


@njit(fastmath=True, cache=True, inline='always')
def morse_potential_inline(r: float, D0: float, alpha: float, r0: float) -> float:
    """Potencial de Morse (inline para eficiencia)."""
    delta_r = r - r0
    exp_term = np.exp(-alpha * delta_r)
    exp2_term = exp_term * exp_term
    return D0 * (exp2_term - 2.0 * exp_term)


@njit(fastmath=True, cache=True)
def compute_total_energy_3d(
    all_positions: np.ndarray,
    atom_types: np.ndarray,
    morse_params_array: np.ndarray
) -> float:
    """
    Calcula la energía total del sistema 3D.

    Args:
        all_positions: Array (N, 3) con posiciones
        atom_types: Array (N,) con tipos
        morse_params_array: Array (3, 3, 3) con parámetros Morse

    Returns:
        Energía total (eV)
    """
    N = len(all_positions)
    energia_total = 0.0

    for i in range(N):
        tipo_i = atom_types[i]
        x_i, y_i, z_i = all_positions[i, 0], all_positions[i, 1], all_positions[i, 2]

        for j in range(i + 1, N):
            tipo_j = atom_types[j]
            x_j, y_j, z_j = all_positions[j, 0], all_positions[j, 1], all_positions[j, 2]

            # Distancia euclidiana 3D
            dx = x_i - x_j
            dy = y_i - y_j
            dz = z_i - z_j
            r = np.sqrt(dx*dx + dy*dy + dz*dz)

            # Parámetros de Morse
            D0 = morse_params_array[tipo_i, tipo_j, 0]
            alpha = morse_params_array[tipo_i, tipo_j, 1]
            r0 = morse_params_array[tipo_i, tipo_j, 2]

            energia_total += morse_potential_inline(r, D0, alpha, r0)

    return energia_total


@njit(fastmath=True, cache=True)
def compute_energy_contribution_3d(
    atom_idx: int,
    all_positions: np.ndarray,
    atom_types: np.ndarray,
    morse_params_array: np.ndarray
) -> float:
    """
    Calcula la contribución a la energía de UN átomo específico.

    Args:
        atom_idx: Índice global del átomo
        all_positions: Array (N, 3) con posiciones
        atom_types: Array (N,) con tipos
        morse_params_array: Array (3, 3, 3)

    Returns:
        Suma de energías de interacción
    """
    N = len(all_positions)
    tipo_atom = atom_types[atom_idx]
    x_atom = all_positions[atom_idx, 0]
    y_atom = all_positions[atom_idx, 1]
    z_atom = all_positions[atom_idx, 2]

    energia = 0.0

    for other_idx in range(N):
        if other_idx == atom_idx:
            continue

        tipo_other = atom_types[other_idx]
        x_other = all_positions[other_idx, 0]
        y_other = all_positions[other_idx, 1]
        z_other = all_positions[other_idx, 2]

        # Distancia 3D
        dx = x_atom - x_other
        dy = y_atom - y_other
        dz = z_atom - z_other
        r = np.sqrt(dx*dx + dy*dy + dz*dz)

        # Parámetros de Morse
        D0 = morse_params_array[tipo_atom, tipo_other, 0]
        alpha = morse_params_array[tipo_atom, tipo_other, 1]
        r0 = morse_params_array[tipo_atom, tipo_other, 2]

        energia += morse_potential_inline(r, D0, alpha, r0)

    return energia


def simulated_annealing_logarithmic_3d(
    all_positions: np.ndarray,
    atom_types: np.ndarray,
    Ti_indices: np.ndarray,
    morse_params_array: np.ndarray,
    c: float = 10000,
    t0: int = 2,
    max_iterations: int = 100000,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Wrapper de alto nivel para Simulated Annealing con enfriamiento logarítmico 3D.

    Args:
        all_positions: Array (112, 3) con todas las posiciones atómicas
        atom_types: Array (112,) con tipos iniciales [0=Fe, 1=Nd, 2=Ti]
        Ti_indices: Array (8,) con índices iniciales de Ti
        morse_params_array: Array (3, 3, 3) con parámetros de Morse
        c: Constante de enfriamiento (default: 10000)
        t0: Offset temporal (default: 2)
        max_iterations: Número de iteraciones (default: 100000)
        seed: Semilla aleatoria (opcional)

    Returns:
        Tupla (atom_types_best, Ti_best, history) donde:
        - atom_types_best: Array (112,) con mejor configuración
        - Ti_best: Array (8,) con índices de Ti en mejor configuración
        - history: Dict con 'energy', 'accepted', 'temperature', 'energy_best'
    """
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)

    atom_types_best, Ti_best, energy_hist, accepted_hist, temp_hist = \
        simulated_annealing_core_logarithmic_3d(
            all_positions, atom_types, Ti_indices, morse_params_array,
            c, t0, max_iterations, seed
        )

    # IMPORTANTE: Recalcular Ti_indices_best desde atom_types_best
    # porque Ti_best del core puede tener índices desactualizados después de muchos swaps
    Ti_indices_best = np.where(atom_types_best == 2)[0].astype(np.int32)

    # Preparar diccionario de historia
    history = {
        'energy': energy_hist,
        'accepted': accepted_hist,
        'temperature': temp_hist,
        'energy_best': np.min(energy_hist)
    }

    return atom_types_best, Ti_indices_best, history
