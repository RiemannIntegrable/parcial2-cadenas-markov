"""
Simulated Annealing optimizado con Numba para el Problema 3 (3D).

Implementa el algoritmo de Recocido Simulado con:
- Movimiento: Swap aleatorio Ti ↔ Fe
- Criterio de aceptación: Metropolis-Hastings
- Cálculo eficiente: ΔE incremental (compute_delta_E_swap_fast_3d)
- Optimización: Numba JIT compilation

El algoritmo logra speedups de 100-1000× vs implementación naive.

Diferencia con punto2: Trabaja con coordenadas 3D en lugar de grilla 2D.
"""

import numpy as np
from numba import njit
from typing import Tuple, Dict, Optional


@njit(fastmath=True)
def simulated_annealing_core_logarithmic_3d(
    all_positions: np.ndarray,
    atom_types: np.ndarray,
    Ti_indices: np.ndarray,
    Fe_indices: np.ndarray,
    morse_params_array: np.ndarray,
    c: float,
    t0: int,
    max_iterations: int,
    seed: int,
    save_every: int = 10
) -> tuple:
    """
    Core del algoritmo de Simulated Annealing con enfriamiento LOGARÍTMICO para 3D.

    Implementa el algoritmo completo usando:
    - Swap aleatorio entre Ti y Fe
    - Metropolis-Hastings para aceptación
    - Cálculo incremental de ΔE (25× más rápido)
    - **ENFRIAMIENTO LOGARÍTMICO**: T(t) = c / log(t + t₀)

    Args:
        all_positions: Array (112, 3) con coordenadas de todos los átomos
        atom_types: Array (112,) con tipos iniciales (0=Fe, 1=Nd, 2=Ti)
        Ti_indices: Array (8,) con índices de los 8 Ti
        Fe_indices: Array (88,) con índices de los 88 Fe
        morse_params_array: Array (3, 3, 3) con parámetros de Morse
        c: Constante de enfriamiento (debe ser ≥ profundidad de barreras)
        t0: Offset temporal (típicamente 2, para evitar log(0))
        max_iterations: Número máximo de iteraciones
        seed: Semilla para reproducibilidad
        save_every: Guardar historia cada N iteraciones (para ahorrar memoria)

    Returns:
        Tupla (atom_types_best, Ti_indices_best, energy_best,
               iterations_to_best, energy_history, accepted_history, temperature_history):
            - atom_types_best: Array (112,) con mejor configuración encontrada
            - Ti_indices_best: Array (8,) con índices de Ti en mejor configuración
            - energy_best: Mejor energía encontrada
            - iterations_to_best: Iteración donde se encontró el óptimo
            - energy_history: Array submuestreado con energías
            - accepted_history: Array submuestreado con aceptaciones
            - temperature_history: Array submuestreado con temperaturas

    Note:
        Este esquema GARANTIZA convergencia al óptimo global (Teorema de Hajek)
        pero requiere MUCHAS iteraciones (típicamente 10⁶-10⁹) para problemas reales.
    """
    # ========================================================================
    # SETUP E INICIALIZACIÓN
    # ========================================================================
    np.random.seed(seed)

    # Estado actual (copias para no modificar originales)
    atom_types_current = atom_types.copy()
    Ti_current = Ti_indices.copy()
    Fe_current = Fe_indices.copy()

    n_Ti = len(Ti_current)
    n_Fe = len(Fe_current)

    # Calcular energía inicial (solo una vez)
    energy_current = 0.0
    n_atoms = len(all_positions)

    # Calcular energía total inicial
    for i in range(n_atoms):
        atom1 = atom_types_current[i]
        x1, y1, z1 = all_positions[i, 0], all_positions[i, 1], all_positions[i, 2]

        for j in range(i + 1, n_atoms):
            atom2 = atom_types_current[j]
            x2, y2, z2 = all_positions[j, 0], all_positions[j, 1], all_positions[j, 2]

            dx = x1 - x2
            dy = y1 - y2
            dz = z1 - z2
            r = np.sqrt(dx * dx + dy * dy + dz * dz)

            D0 = morse_params_array[atom1, atom2, 0]
            alpha_param = morse_params_array[atom1, atom2, 1]
            r0 = morse_params_array[atom1, atom2, 2]

            delta_r = r - r0
            exp_term = np.exp(-alpha_param * delta_r)
            exp2_term = exp_term * exp_term
            U = D0 * (exp2_term - 2.0 * exp_term)
            energy_current += U

    # Mejor estado encontrado
    atom_types_best = atom_types_current.copy()
    Ti_best = Ti_current.copy()
    energy_best = energy_current
    iterations_to_best = 0

    # Historia (submuestreada para ahorrar memoria)
    history_size = max_iterations // save_every
    energy_history = np.zeros(history_size, dtype=np.float64)
    accepted_history = np.zeros(history_size, dtype=np.bool_)
    temperature_history = np.zeros(history_size, dtype=np.float64)

    # ========================================================================
    # LOOP PRINCIPAL DE SIMULATED ANNEALING
    # ========================================================================
    for iteration in range(max_iterations):
        # Calcular temperatura (enfriamiento LOGARÍTMICO)
        # T(t) = c / log(t + t₀)
        log_val = np.log(float(iteration + t0))
        if log_val < 0.1:  # Evitar divisiones por valores muy pequeños
            log_val = 0.1
        T = c / log_val

        # ====================================================================
        # PROPONER MOVIMIENTO: Swap Ti ↔ Fe
        # ====================================================================
        ti_local_idx = np.random.randint(0, n_Ti)
        fe_local_idx = np.random.randint(0, n_Fe)

        ti_global_idx = Ti_current[ti_local_idx]
        fe_global_idx = Fe_current[fe_local_idx]

        # ====================================================================
        # CALCULAR ΔE INCREMENTAL (OPTIMIZACIÓN CRÍTICA)
        # ====================================================================
        # Inline de compute_delta_E_swap_fast_3d para Numba
        ti_x = all_positions[ti_global_idx, 0]
        ti_y = all_positions[ti_global_idx, 1]
        ti_z = all_positions[ti_global_idx, 2]
        fe_x = all_positions[fe_global_idx, 0]
        fe_y = all_positions[fe_global_idx, 1]
        fe_z = all_positions[fe_global_idx, 2]

        delta_E = 0.0

        for k in range(n_atoms):
            if k == ti_global_idx or k == fe_global_idx:
                continue

            atom_other = atom_types_current[k]
            other_x = all_positions[k, 0]
            other_y = all_positions[k, 1]
            other_z = all_positions[k, 2]

            # Energía que perdemos
            dx_old_Ti = ti_x - other_x
            dy_old_Ti = ti_y - other_y
            dz_old_Ti = ti_z - other_z
            r_old_Ti = np.sqrt(dx_old_Ti * dx_old_Ti + dy_old_Ti * dy_old_Ti + dz_old_Ti * dz_old_Ti)
            D0_Ti = morse_params_array[2, atom_other, 0]
            alpha_Ti = morse_params_array[2, atom_other, 1]
            r0_Ti = morse_params_array[2, atom_other, 2]
            delta_r = r_old_Ti - r0_Ti
            exp_term = np.exp(-alpha_Ti * delta_r)
            exp2_term = exp_term * exp_term
            U_old_Ti = D0_Ti * (exp2_term - 2.0 * exp_term)

            dx_old_Fe = fe_x - other_x
            dy_old_Fe = fe_y - other_y
            dz_old_Fe = fe_z - other_z
            r_old_Fe = np.sqrt(dx_old_Fe * dx_old_Fe + dy_old_Fe * dy_old_Fe + dz_old_Fe * dz_old_Fe)
            D0_Fe = morse_params_array[0, atom_other, 0]
            alpha_Fe = morse_params_array[0, atom_other, 1]
            r0_Fe = morse_params_array[0, atom_other, 2]
            delta_r = r_old_Fe - r0_Fe
            exp_term = np.exp(-alpha_Fe * delta_r)
            exp2_term = exp_term * exp_term
            U_old_Fe = D0_Fe * (exp2_term - 2.0 * exp_term)

            # Energía que ganamos
            dx_new_Ti = fe_x - other_x
            dy_new_Ti = fe_y - other_y
            dz_new_Ti = fe_z - other_z
            r_new_Ti = np.sqrt(dx_new_Ti * dx_new_Ti + dy_new_Ti * dy_new_Ti + dz_new_Ti * dz_new_Ti)
            delta_r = r_new_Ti - r0_Ti
            exp_term = np.exp(-alpha_Ti * delta_r)
            exp2_term = exp_term * exp_term
            U_new_Ti = D0_Ti * (exp2_term - 2.0 * exp_term)

            dx_new_Fe = ti_x - other_x
            dy_new_Fe = ti_y - other_y
            dz_new_Fe = ti_z - other_z
            r_new_Fe = np.sqrt(dx_new_Fe * dx_new_Fe + dy_new_Fe * dy_new_Fe + dz_new_Fe * dz_new_Fe)
            delta_r = r_new_Fe - r0_Fe
            exp_term = np.exp(-alpha_Fe * delta_r)
            exp2_term = exp_term * exp_term
            U_new_Fe = D0_Fe * (exp2_term - 2.0 * exp_term)

            delta_E += (U_new_Ti + U_new_Fe - U_old_Ti - U_old_Fe)

        # ====================================================================
        # CRITERIO DE METROPOLIS-HASTINGS
        # ====================================================================
        accept = False

        if delta_E < 0:
            # Mejora: siempre aceptar
            accept = True
        elif T > 1e-10:
            # Empeora: aceptar con probabilidad exp(-ΔE/T)
            prob_aceptacion = np.exp(-delta_E / T)
            if np.random.random() < prob_aceptacion:
                accept = True

        # ====================================================================
        # APLICAR MOVIMIENTO SI SE ACEPTA
        # ====================================================================
        if accept:
            # Swap en atom_types
            atom_types_current[ti_global_idx] = 0  # Fe
            atom_types_current[fe_global_idx] = 2  # Ti

            # Actualizar arrays de índices
            Ti_current[ti_local_idx] = fe_global_idx
            Fe_current[fe_local_idx] = ti_global_idx

            # Actualizar energía (incremental)
            energy_current += delta_E

            # Actualizar mejor si corresponde
            if energy_current < energy_best:
                energy_best = energy_current
                atom_types_best = atom_types_current.copy()
                Ti_best = Ti_current.copy()
                iterations_to_best = iteration

        # ====================================================================
        # GUARDAR HISTORIA (submuestreada)
        # ====================================================================
        if iteration % save_every == 0:
            hist_idx = iteration // save_every
            if hist_idx < history_size:
                energy_history[hist_idx] = energy_current
                accepted_history[hist_idx] = accept
                temperature_history[hist_idx] = T

    # ========================================================================
    # RETORNAR RESULTADOS
    # ========================================================================
    return (atom_types_best, Ti_best, energy_best, iterations_to_best,
            energy_history, accepted_history, temperature_history)


# ============================================================================
# WRAPPER PYTHON (NO-NUMBA) PARA FACILIDAD DE USO
# ============================================================================

def simulated_annealing_logarithmic_3d(
    all_positions: np.ndarray,
    atom_types: np.ndarray,
    Ti_indices: np.ndarray,
    Fe_indices: np.ndarray,
    morse_params_array: np.ndarray,
    c: float,
    t0: int = 2,
    max_iterations: int = 1000000,
    seed: Optional[int] = None,
    save_every: int = 10
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Simulated Annealing con ENFRIAMIENTO LOGARÍTMICO para estructura 3D.

    Este esquema garantiza convergencia al óptimo global (Teorema de Hajek)
    pero requiere MUCHAS iteraciones para problemas reales.

    T(t) = c / log(t + t₀)

    Args:
        all_positions: Array (112, 3) con coordenadas de todos los átomos
        atom_types: Array (112,) con tipos iniciales
        Ti_indices: Array (8,) con índices iniciales de Ti
        Fe_indices: Array (88,) con índices iniciales de Fe
        morse_params_array: Array (3, 3, 3) con parámetros de Morse
        c: Constante de enfriamiento (debe ser ≥ profundidad de barreras Δ)
        t0: Offset temporal (típicamente 2, para evitar log(0))
        max_iterations: Número de iteraciones (típicamente 10⁶-10⁹ para garantías)
        seed: Semilla para reproducibilidad (opcional)
        save_every: Guardar historia cada N iteraciones (default 10)

    Returns:
        Tupla (atom_types_best, Ti_indices_best, history):
            - atom_types_best: Array (112,) con mejor configuración
            - Ti_indices_best: Array (8,) con índices de Ti en mejor configuración
            - history: Dict con claves:
                - 'energy': Array submuestreado de energías
                - 'accepted': Array submuestreado booleano de aceptación
                - 'temperature': Array submuestreado de temperaturas
                - 'energy_best': Mejor energía encontrada
                - 'iterations_to_best': Iteración donde se encontró

    Examples:
        >>> from src.punto3.crystal import crear_configuracion_inicial
        >>> from src.punto3.morse import preparar_morse_params_array
        >>> all_pos, types, Ti_idx, Fe_idx, _ = crear_configuracion_inicial(seed=42)
        >>> params = preparar_morse_params_array()
        >>> types_opt, Ti_opt, history = simulated_annealing_logarithmic_3d(
        ...     all_pos, types, Ti_idx, Fe_idx, params,
        ...     c=10000, t0=2, max_iterations=1000000, seed=42
        ... )
        >>> history['energy_best'] <= history['energy'][0]  # Mejoró
        True

    Note:
        **ADVERTENCIA**: Este esquema requiere iteraciones del orden de 10⁶-10⁹
        para garantizar convergencia. Para aplicaciones prácticas, considerar
        usar múltiples runs independientes.

        Referencia: Hajek, B. (1988). "Cooling Schedules for Optimal Annealing"
    """
    # Usar semilla por defecto si no se proporciona
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)

    # Llamar al core optimizado con enfriamiento logarítmico
    (atom_types_best, Ti_best, energy_best, iterations_to_best,
     energy_hist, accepted_hist, temp_hist) = simulated_annealing_core_logarithmic_3d(
        all_positions,
        atom_types,
        Ti_indices,
        Fe_indices,
        morse_params_array,
        c,
        t0,
        max_iterations,
        seed,
        save_every
    )

    # Preparar diccionario de historia
    history = {
        'energy': energy_hist,
        'accepted': accepted_hist,
        'temperature': temp_hist,
        'energy_best': energy_best,
        'iterations_to_best': iterations_to_best
    }

    return atom_types_best, Ti_best, history
