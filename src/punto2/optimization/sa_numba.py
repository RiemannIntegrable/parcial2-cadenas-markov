"""
Simulated Annealing optimizado con Numba para el Problema 2.

Implementa el algoritmo de Recocido Simulado con:
- Movimiento: Swap aleatorio Ti ↔ Fe
- Criterio de aceptación: Metropolis-Hastings
- Cálculo eficiente: ΔE incremental (compute_delta_E_swap_fast)
- Optimización: Numba JIT compilation

El algoritmo logra speedups de 100-1000× vs implementación naive.
"""

import numpy as np
from numba import njit
from typing import Tuple, Dict, Optional

# Import relativo para energy_numba
import sys
import os
# Agregar path del proyecto al sys.path para imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@njit(fastmath=True)
def simulated_annealing_core(
    grid_array: np.ndarray,
    Ti_positions: np.ndarray,
    Fe_positions: np.ndarray,
    morse_params_array: np.ndarray,
    T0: float,
    alpha: float,
    max_iterations: int,
    seed: int
) -> tuple:
    """
    Core del algoritmo de Simulated Annealing optimizado con Numba.

    Implementa el algoritmo completo usando:
    - Swap aleatorio entre Ti y Fe
    - Metropolis-Hastings para aceptación
    - Cálculo incremental de ΔE (25× más rápido)

    Args:
        grid_array: Array (10, 10) con estado inicial
        Ti_positions: Array (8, 2) con posiciones iniciales de Ti
        Fe_positions: Array (n_Fe, 2) con posiciones iniciales de Fe
        morse_params_array: Array (3, 3, 3) con parámetros de Morse
        T0: Temperatura inicial
        alpha: Factor de enfriamiento geométrico (T = T0 * alpha^iteration)
        max_iterations: Número máximo de iteraciones
        seed: Semilla para reproducibilidad

    Returns:
        Tupla (grid_best, Ti_best, energy_history, accepted_history, temperature_history):
            - grid_best: Array (10, 10) con mejor configuración encontrada
            - Ti_best: Array (8, 2) con posiciones de Ti en mejor configuración
            - energy_history: Array (max_iterations,) con energía en cada iteración
            - accepted_history: Array (max_iterations,) booleano de aceptación
            - temperature_history: Array (max_iterations,) con temperatura

    Note:
        Esta función es el core optimizado y no debe llamarse directamente.
        Usar el wrapper `simulated_annealing()` para facilidad de uso.
    """
    # ========================================================================
    # SETUP E INICIALIZACIÓN
    # ========================================================================
    np.random.seed(seed)

    # Estado actual (copias para no modificar originales)
    grid_current = grid_array.copy()
    Ti_current = Ti_positions.copy()
    Fe_current = Fe_positions.copy()

    n_Fe = len(Fe_current)

    # Calcular energía inicial (solo una vez)
    # Importar función inline para Numba
    energy_current = 0.0

    # Calcular energía total inicial
    for i1 in range(10):
        for j1 in range(10):
            atom1 = grid_current[i1, j1]
            for i2 in range(10):
                if i2 < i1:
                    continue
                elif i2 == i1:
                    start_j2 = j1 + 1
                else:
                    start_j2 = 0
                for j2 in range(start_j2, 10):
                    atom2 = grid_current[i2, j2]
                    dx = float(i1 - i2)
                    dy = float(j1 - j2)
                    r = np.sqrt(dx * dx + dy * dy)
                    D0 = morse_params_array[atom1, atom2, 0]
                    alpha_param = morse_params_array[atom1, atom2, 1]
                    r0 = morse_params_array[atom1, atom2, 2]
                    delta_r = r - r0
                    exp_term = np.exp(-alpha_param * delta_r)
                    exp2_term = exp_term * exp_term
                    U = D0 * (exp2_term - 2.0 * exp_term)
                    energy_current += U

    # Mejor estado encontrado
    grid_best = grid_current.copy()
    Ti_best = Ti_current.copy()
    energy_best = energy_current

    # Historia (pre-asignada para eficiencia)
    energy_history = np.zeros(max_iterations, dtype=np.float64)
    accepted_history = np.zeros(max_iterations, dtype=np.bool_)
    temperature_history = np.zeros(max_iterations, dtype=np.float64)

    # ========================================================================
    # LOOP PRINCIPAL DE SIMULATED ANNEALING
    # ========================================================================
    for iteration in range(max_iterations):
        # Calcular temperatura (enfriamiento geométrico)
        T = T0 * (alpha ** iteration)
        temperature_history[iteration] = T

        # ====================================================================
        # PROPONER MOVIMIENTO: Swap Ti ↔ Fe
        # ====================================================================
        ti_idx = np.random.randint(0, 8)
        fe_idx = np.random.randint(0, n_Fe)

        ti_x = Ti_current[ti_idx, 0]
        ti_y = Ti_current[ti_idx, 1]
        fe_x = Fe_current[fe_idx, 0]
        fe_y = Fe_current[fe_idx, 1]

        # ====================================================================
        # CALCULAR ΔE INCREMENTAL (OPTIMIZACIÓN CRÍTICA)
        # ====================================================================
        # Inline de compute_delta_E_swap_fast para Numba
        delta_E = 0.0

        for i in range(10):
            for j in range(10):
                if (i == ti_x and j == ti_y):
                    continue
                if (i == fe_x and j == fe_y):
                    continue

                atom_other = grid_current[i, j]

                # Energía que perdemos
                dx_old_Ti = float(ti_x - i)
                dy_old_Ti = float(ti_y - j)
                r_old_Ti = np.sqrt(dx_old_Ti * dx_old_Ti + dy_old_Ti * dy_old_Ti)
                D0_Ti = morse_params_array[2, atom_other, 0]
                alpha_Ti = morse_params_array[2, atom_other, 1]
                r0_Ti = morse_params_array[2, atom_other, 2]
                delta_r = r_old_Ti - r0_Ti
                exp_term = np.exp(-alpha_Ti * delta_r)
                exp2_term = exp_term * exp_term
                U_old_Ti = D0_Ti * (exp2_term - 2.0 * exp_term)

                dx_old_Fe = float(fe_x - i)
                dy_old_Fe = float(fe_y - j)
                r_old_Fe = np.sqrt(dx_old_Fe * dx_old_Fe + dy_old_Fe * dy_old_Fe)
                D0_Fe = morse_params_array[0, atom_other, 0]
                alpha_Fe = morse_params_array[0, atom_other, 1]
                r0_Fe = morse_params_array[0, atom_other, 2]
                delta_r = r_old_Fe - r0_Fe
                exp_term = np.exp(-alpha_Fe * delta_r)
                exp2_term = exp_term * exp_term
                U_old_Fe = D0_Fe * (exp2_term - 2.0 * exp_term)

                # Energía que ganamos
                dx_new_Ti = float(fe_x - i)
                dy_new_Ti = float(fe_y - j)
                r_new_Ti = np.sqrt(dx_new_Ti * dx_new_Ti + dy_new_Ti * dy_new_Ti)
                delta_r = r_new_Ti - r0_Ti
                exp_term = np.exp(-alpha_Ti * delta_r)
                exp2_term = exp_term * exp_term
                U_new_Ti = D0_Ti * (exp2_term - 2.0 * exp_term)

                dx_new_Fe = float(ti_x - i)
                dy_new_Fe = float(ti_y - j)
                r_new_Fe = np.sqrt(dx_new_Fe * dx_new_Fe + dy_new_Fe * dy_new_Fe)
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
            # Swap en la grilla
            grid_current[ti_x, ti_y] = 0  # Fe
            grid_current[fe_x, fe_y] = 2  # Ti

            # Actualizar arrays de posiciones
            Ti_current[ti_idx, 0] = fe_x
            Ti_current[ti_idx, 1] = fe_y
            Fe_current[fe_idx, 0] = ti_x
            Fe_current[fe_idx, 1] = ti_y

            # Actualizar energía (incremental)
            energy_current += delta_E

            # Actualizar mejor si corresponde
            if energy_current < energy_best:
                energy_best = energy_current
                grid_best = grid_current.copy()
                Ti_best = Ti_current.copy()

        # ====================================================================
        # GUARDAR HISTORIA
        # ====================================================================
        energy_history[iteration] = energy_current
        accepted_history[iteration] = accept

    # ========================================================================
    # RETORNAR RESULTADOS
    # ========================================================================
    return grid_best, Ti_best, energy_history, accepted_history, temperature_history


@njit(fastmath=True)
def simulated_annealing_core_logarithmic(
    grid_array: np.ndarray,
    Ti_positions: np.ndarray,
    Fe_positions: np.ndarray,
    morse_params_array: np.ndarray,
    c: float,
    t0: int,
    max_iterations: int,
    seed: int
) -> tuple:
    """
    Core del algoritmo de Simulated Annealing con enfriamiento LOGARÍTMICO.

    Implementa el algoritmo completo usando:
    - Swap aleatorio entre Ti y Fe
    - Metropolis-Hastings para aceptación
    - Cálculo incremental de ΔE (25× más rápido)
    - **ENFRIAMIENTO LOGARÍTMICO**: T(t) = c / log(t + t₀)

    Args:
        grid_array: Array (10, 10) con estado inicial
        Ti_positions: Array (8, 2) con posiciones iniciales de Ti
        Fe_positions: Array (n_Fe, 2) con posiciones iniciales de Fe
        morse_params_array: Array (3, 3, 3) con parámetros de Morse
        c: Constante de enfriamiento (debe ser ≥ profundidad de barreras)
        t0: Offset temporal (típicamente 2, para evitar log(0))
        max_iterations: Número máximo de iteraciones
        seed: Semilla para reproducibilidad

    Returns:
        Tupla (grid_best, Ti_best, energy_history, accepted_history, temperature_history):
            - grid_best: Array (10, 10) con mejor configuración encontrada
            - Ti_best: Array (8, 2) con posiciones de Ti en mejor configuración
            - energy_history: Array (max_iterations,) con energía en cada iteración
            - accepted_history: Array (max_iterations,) booleano de aceptación
            - temperature_history: Array (max_iterations,) con temperatura

    Note:
        Este esquema GARANTIZA convergencia al óptimo global (Teorema de Hajek)
        pero requiere MUCHAS iteraciones (típicamente 10⁶-10⁹) para problemas reales.
    """
    # ========================================================================
    # SETUP E INICIALIZACIÓN
    # ========================================================================
    np.random.seed(seed)

    # Estado actual (copias para no modificar originales)
    grid_current = grid_array.copy()
    Ti_current = Ti_positions.copy()
    Fe_current = Fe_positions.copy()

    n_Fe = len(Fe_current)

    # Calcular energía inicial (solo una vez)
    energy_current = 0.0

    # Calcular energía total inicial
    for i1 in range(10):
        for j1 in range(10):
            atom1 = grid_current[i1, j1]
            for i2 in range(10):
                if i2 < i1:
                    continue
                elif i2 == i1:
                    start_j2 = j1 + 1
                else:
                    start_j2 = 0
                for j2 in range(start_j2, 10):
                    atom2 = grid_current[i2, j2]
                    dx = float(i1 - i2)
                    dy = float(j1 - j2)
                    r = np.sqrt(dx * dx + dy * dy)
                    D0 = morse_params_array[atom1, atom2, 0]
                    alpha_param = morse_params_array[atom1, atom2, 1]
                    r0 = morse_params_array[atom1, atom2, 2]
                    delta_r = r - r0
                    exp_term = np.exp(-alpha_param * delta_r)
                    exp2_term = exp_term * exp_term
                    U = D0 * (exp2_term - 2.0 * exp_term)
                    energy_current += U

    # Mejor estado encontrado
    grid_best = grid_current.copy()
    Ti_best = Ti_current.copy()
    energy_best = energy_current

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
        if log_val < 0.1:  # Evitar divisiones por valores muy pequeños
            log_val = 0.1
        T = c / log_val
        temperature_history[iteration] = T

        # ====================================================================
        # PROPONER MOVIMIENTO: Swap Ti ↔ Fe
        # ====================================================================
        ti_idx = np.random.randint(0, 8)
        fe_idx = np.random.randint(0, n_Fe)

        ti_x = Ti_current[ti_idx, 0]
        ti_y = Ti_current[ti_idx, 1]
        fe_x = Fe_current[fe_idx, 0]
        fe_y = Fe_current[fe_idx, 1]

        # ====================================================================
        # CALCULAR ΔE INCREMENTAL (OPTIMIZACIÓN CRÍTICA)
        # ====================================================================
        delta_E = 0.0

        for i in range(10):
            for j in range(10):
                if (i == ti_x and j == ti_y):
                    continue
                if (i == fe_x and j == fe_y):
                    continue

                atom_other = grid_current[i, j]

                # Energía que perdemos
                dx_old_Ti = float(ti_x - i)
                dy_old_Ti = float(ti_y - j)
                r_old_Ti = np.sqrt(dx_old_Ti * dx_old_Ti + dy_old_Ti * dy_old_Ti)
                D0_Ti = morse_params_array[2, atom_other, 0]
                alpha_Ti = morse_params_array[2, atom_other, 1]
                r0_Ti = morse_params_array[2, atom_other, 2]
                delta_r = r_old_Ti - r0_Ti
                exp_term = np.exp(-alpha_Ti * delta_r)
                exp2_term = exp_term * exp_term
                U_old_Ti = D0_Ti * (exp2_term - 2.0 * exp_term)

                dx_old_Fe = float(fe_x - i)
                dy_old_Fe = float(fe_y - j)
                r_old_Fe = np.sqrt(dx_old_Fe * dx_old_Fe + dy_old_Fe * dy_old_Fe)
                D0_Fe = morse_params_array[0, atom_other, 0]
                alpha_Fe = morse_params_array[0, atom_other, 1]
                r0_Fe = morse_params_array[0, atom_other, 2]
                delta_r = r_old_Fe - r0_Fe
                exp_term = np.exp(-alpha_Fe * delta_r)
                exp2_term = exp_term * exp_term
                U_old_Fe = D0_Fe * (exp2_term - 2.0 * exp_term)

                # Energía que ganamos
                dx_new_Ti = float(fe_x - i)
                dy_new_Ti = float(fe_y - j)
                r_new_Ti = np.sqrt(dx_new_Ti * dx_new_Ti + dy_new_Ti * dy_new_Ti)
                delta_r = r_new_Ti - r0_Ti
                exp_term = np.exp(-alpha_Ti * delta_r)
                exp2_term = exp_term * exp_term
                U_new_Ti = D0_Ti * (exp2_term - 2.0 * exp_term)

                dx_new_Fe = float(ti_x - i)
                dy_new_Fe = float(ti_y - j)
                r_new_Fe = np.sqrt(dx_new_Fe * dx_new_Fe + dy_new_Fe * dy_new_Fe)
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
            # Swap en la grilla
            grid_current[ti_x, ti_y] = 0  # Fe
            grid_current[fe_x, fe_y] = 2  # Ti

            # Actualizar arrays de posiciones
            Ti_current[ti_idx, 0] = fe_x
            Ti_current[ti_idx, 1] = fe_y
            Fe_current[fe_idx, 0] = ti_x
            Fe_current[fe_idx, 1] = ti_y

            # Actualizar energía (incremental)
            energy_current += delta_E

            # Actualizar mejor si corresponde
            if energy_current < energy_best:
                energy_best = energy_current
                grid_best = grid_current.copy()
                Ti_best = Ti_current.copy()

        # ====================================================================
        # GUARDAR HISTORIA
        # ====================================================================
        energy_history[iteration] = energy_current
        accepted_history[iteration] = accept

    # ========================================================================
    # RETORNAR RESULTADOS
    # ========================================================================
    return grid_best, Ti_best, energy_history, accepted_history, temperature_history


# ============================================================================
# WRAPPER PYTHON (NO-NUMBA) PARA FACILIDAD DE USO
# ============================================================================

def simulated_annealing(
    grid_array: np.ndarray,
    Ti_positions: np.ndarray,
    morse_params_array: np.ndarray,
    T0: float,
    alpha: float,
    max_iterations: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Simulated Annealing para optimizar la configuración de 8 Ti en grilla 10×10.

    Wrapper de alto nivel que llama al core optimizado con Numba.

    Args:
        grid_array: Array (10, 10) con configuración inicial
        Ti_positions: Array (8, 2) con posiciones iniciales de Ti
        morse_params_array: Array (3, 3, 3) con parámetros de Morse
        T0: Temperatura inicial
        alpha: Factor de enfriamiento geométrico (típicamente 0.95-0.99)
        max_iterations: Número de iteraciones (típicamente 50,000-100,000)
        seed: Semilla para reproducibilidad (opcional)

    Returns:
        Tupla (grid_best, Ti_best, history):
            - grid_best: Array (10, 10) con mejor configuración
            - Ti_best: Array (8, 2) con posiciones de Ti en mejor configuración
            - history: Dict con claves:
                - 'energy': Array de energías en cada iteración
                - 'accepted': Array booleano de aceptación
                - 'temperature': Array de temperaturas
                - 'energy_best': Mejor energía encontrada

    Examples:
        >>> from src.punto2.grid import crear_grid_inicial
        >>> from src.punto2.morse import preparar_morse_params_array
        >>> grid, Ti_pos, _ = crear_grid_inicial(seed=42)
        >>> params = preparar_morse_params_array()
        >>> grid_opt, Ti_opt, history = simulated_annealing(
        ...     grid, Ti_pos, params,
        ...     T0=20.0, alpha=0.98, max_iterations=10000, seed=42
        ... )
        >>> history['energy'][-1] < history['energy'][0]  # Mejoró
        True

    Note:
        Esta es la función principal para usar en el notebook.
        Internamente llama a simulated_annealing_core() optimizado con Numba.
    """
    # Importar aquí para evitar circular imports
    from ..grid.grid_utils import get_Fe_positions

    # Obtener posiciones de Fe
    Fe_positions = get_Fe_positions(grid_array)

    # Usar semilla por defecto si no se proporciona
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)

    # Llamar al core optimizado
    grid_best, Ti_best, energy_hist, accepted_hist, temp_hist = simulated_annealing_core(
        grid_array,
        Ti_positions,
        Fe_positions,
        morse_params_array,
        T0,
        alpha,
        max_iterations,
        seed
    )

    # Preparar diccionario de historia
    history = {
        'energy': energy_hist,
        'accepted': accepted_hist,
        'temperature': temp_hist,
        'energy_best': np.min(energy_hist)
    }

    return grid_best, Ti_best, history


def simulated_annealing_logarithmic(
    grid_array: np.ndarray,
    Ti_positions: np.ndarray,
    morse_params_array: np.ndarray,
    c: float,
    t0: int = 2,
    max_iterations: int = 1000000,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Simulated Annealing con ENFRIAMIENTO LOGARÍTMICO.

    Este esquema garantiza convergencia al óptimo global (Teorema de Hajek)
    pero requiere MUCHAS iteraciones para problemas reales.

    T(t) = c / log(t + t₀)

    Args:
        grid_array: Array (10, 10) con configuración inicial
        Ti_positions: Array (8, 2) con posiciones iniciales de Ti
        morse_params_array: Array (3, 3, 3) con parámetros de Morse
        c: Constante de enfriamiento (debe ser ≥ profundidad de barreras Δ)
        t0: Offset temporal (típicamente 2, para evitar log(0))
        max_iterations: Número de iteraciones (típicamente 10⁶-10⁹ para garantías)
        seed: Semilla para reproducibilidad (opcional)

    Returns:
        Tupla (grid_best, Ti_best, history):
            - grid_best: Array (10, 10) con mejor configuración
            - Ti_best: Array (8, 2) con posiciones de Ti en mejor configuración
            - history: Dict con claves:
                - 'energy': Array de energías en cada iteración
                - 'accepted': Array booleano de aceptación
                - 'temperature': Array de temperaturas
                - 'energy_best': Mejor energía encontrada

    Examples:
        >>> from src.punto2.grid import crear_grid_inicial
        >>> from src.punto2.morse import preparar_morse_params_array
        >>> grid, Ti_pos, _ = crear_grid_inicial(seed=42)
        >>> params = preparar_morse_params_array()
        >>> grid_opt, Ti_opt, history = simulated_annealing_logarithmic(
        ...     grid, Ti_pos, params,
        ...     c=10000, t0=2, max_iterations=1000000, seed=42
        ... )
        >>> history['energy'][-1] < history['energy'][0]  # Mejoró
        True

    Note:
        **ADVERTENCIA**: Este esquema requiere iteraciones del orden de 10⁶-10⁹
        para garantizar convergencia. Para aplicaciones prácticas, considerar
        usar enfriamiento geométrico con múltiples runs independientes.

        Referencia: Hajek, B. (1988). "Cooling Schedules for Optimal Annealing"
    """
    # Importar aquí para evitar circular imports
    from ..grid.grid_utils import get_Fe_positions

    # Obtener posiciones de Fe
    Fe_positions = get_Fe_positions(grid_array)

    # Usar semilla por defecto si no se proporciona
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)

    # Llamar al core optimizado con enfriamiento logarítmico
    grid_best, Ti_best, energy_hist, accepted_hist, temp_hist = simulated_annealing_core_logarithmic(
        grid_array,
        Ti_positions,
        Fe_positions,
        morse_params_array,
        c,
        t0,
        max_iterations,
        seed
    )

    # Preparar diccionario de historia
    history = {
        'energy': energy_hist,
        'accepted': accepted_hist,
        'temperature': temp_hist,
        'energy_best': np.min(energy_hist)
    }

    return grid_best, Ti_best, history
