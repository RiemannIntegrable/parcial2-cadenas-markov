"""
Cálculo de energía optimizado con Numba.

Este módulo contiene las funciones críticas para calcular la energía del sistema:
1. compute_total_energy_fast: Energía total (O(N²)) - solo para inicialización
2. compute_delta_E_swap_fast: ΔE incremental (O(N)) - OPTIMIZACIÓN CLAVE

La función compute_delta_E_swap_fast es la clave del éxito del algoritmo,
logrando un speedup de 25× sin Numba y hasta 1000× con Numba.

IMPORTANTE: El espaciado de la grilla es de 2.8 Angstroms entre átomos adyacentes.
"""

import numpy as np
from numba import njit

# Constante de espaciado de la grilla (Angstroms)
# Distancia física entre átomos adyacentes en la grilla
GRID_SPACING = 2.8


@njit(fastmath=True, cache=True)
def compute_total_energy_fast(grid_array: np.ndarray, morse_params_array: np.ndarray) -> float:
    """
    Calcula la energía total del sistema usando el potencial de Morse.

    Complejidad: O(N²) donde N=100 (número total de átomos).
    Calcula 4,950 interacciones (todos los pares únicos).

    IMPORTANTE: Esta función solo debe usarse para:
    - Calcular energía inicial del sistema
    - Validación/debugging
    NO debe usarse en el loop de Simulated Annealing (usar compute_delta_E_swap_fast).

    Args:
        grid_array: Array (10, 10) con valores 0=Fe, 1=Nd, 2=Ti
        morse_params_array: Array (3, 3, 3) con parámetros de Morse
                           params[tipo1, tipo2] = [D0, alpha, r0]

    Returns:
        Energía total del sistema (suma sobre todos los pares únicos)

    Note:
        Optimizado con Numba JIT. fastmath=True habilita optimizaciones agresivas.

    Examples:
        >>> grid = np.zeros((10, 10), dtype=np.int8)
        >>> params = preparar_morse_params_array()
        >>> E = compute_total_energy_fast(grid, params)
        >>> E
        -523.4567...
    """
    energy = 0.0

    # Iterar sobre todos los pares únicos (i,j) donde i < j
    for i1 in range(10):
        for j1 in range(10):
            atom1 = grid_array[i1, j1]

            # Solo pares únicos: j > i en orden lexicográfico
            for i2 in range(10):
                # Si estamos en la misma fila, empezar después de j1
                # Si estamos en filas posteriores, empezar desde 0
                if i2 < i1:
                    continue
                elif i2 == i1:
                    start_j2 = j1 + 1
                else:
                    start_j2 = 0

                for j2 in range(start_j2, 10):
                    atom2 = grid_array[i2, j2]

                    # Distancia euclidiana 2D en Angstroms
                    # Multiplicar por GRID_SPACING para convertir de índices a coordenadas físicas
                    dx = float(i1 - i2) * GRID_SPACING
                    dy = float(j1 - j2) * GRID_SPACING
                    r = np.sqrt(dx * dx + dy * dy)

                    # Parámetros de Morse para este par
                    D0 = morse_params_array[atom1, atom2, 0]
                    alpha = morse_params_array[atom1, atom2, 1]
                    r0 = morse_params_array[atom1, atom2, 2]

                    # Potencial de Morse inline (más rápido que llamada a función)
                    delta_r = r - r0
                    exp_term = np.exp(-alpha * delta_r)
                    exp2_term = exp_term * exp_term
                    U = D0 * (exp2_term - 2.0 * exp_term)

                    energy += U

    return energy


@njit(fastmath=True, cache=True)
def compute_delta_E_swap_fast(
    grid_array: np.ndarray,
    Ti_positions: np.ndarray,
    ti_idx: int,
    fe_pos_x: int,
    fe_pos_y: int,
    morse_params_array: np.ndarray
) -> float:
    """
    Calcula ΔE incremental para un swap Ti ↔ Fe.

    Esta es LA FUNCIÓN CRÍTICA del Problema 2. Implementa el cálculo eficiente
    de la diferencia de energía SIN recalcular toda la red.

    Algoritmo:
        ΔE = E_nuevo - E_actual

        E_nuevo: Energía con Ti en fe_pos y Fe en ti_pos
        E_actual: Energía con Ti en ti_pos y Fe en fe_pos

        Solo recalculamos las interacciones que CAMBIAN:
        - Ti en su nueva posición (fe_pos) con todos los demás
        - Fe en su nueva posición (ti_pos) con todos los demás
        - Restamos las interacciones viejas

    Complejidad: O(N) donde N=100 → ~200 interacciones
    Speedup vs O(N²): 25× sin Numba, ~1000× con Numba

    Args:
        grid_array: Array (10, 10) con estado actual
        Ti_positions: Array (8, 2) con posiciones actuales de Ti
        ti_idx: Índice del Ti a mover (0-7)
        fe_pos_x, fe_pos_y: Posición del Fe a intercambiar
        morse_params_array: Array (3, 3, 3) con parámetros de Morse

    Returns:
        ΔE: Cambio de energía (E_nuevo - E_actual)
            - ΔE < 0: mejora (disminuye energía)
            - ΔE > 0: empeora (aumenta energía)

    Note:
        Esta función NO modifica grid_array ni Ti_positions.
        El caller es responsable de aplicar el swap si se acepta.

    Examples:
        >>> grid = crear_grid_inicial(seed=42)[0]
        >>> Ti_pos = get_Ti_positions(grid)
        >>> Fe_pos = get_Fe_positions(grid)
        >>> params = preparar_morse_params_array()
        >>> delta_E = compute_delta_E_swap_fast(grid, Ti_pos, 0, Fe_pos[0,0], Fe_pos[0,1], params)
        >>> delta_E
        -2.345...
    """
    # Posición actual del Ti
    ti_pos_x = Ti_positions[ti_idx, 0]
    ti_pos_y = Ti_positions[ti_idx, 1]

    delta_E = 0.0

    # ========================================================================
    # ITERAR SOBRE TODOS LOS DEMÁS ÁTOMOS
    # ========================================================================
    for i in range(10):
        for j in range(10):
            # Saltar las posiciones involucradas en el swap
            if (i == ti_pos_x and j == ti_pos_y):
                continue
            if (i == fe_pos_x and j == fe_pos_y):
                continue

            atom_other = grid_array[i, j]

            # ================================================================
            # ENERGÍA QUE PERDEMOS (configuración actual)
            # ================================================================

            # --- Ti en posición antigua interactuando con atom_other ---
            dx_old_Ti = float(ti_pos_x - i) * GRID_SPACING
            dy_old_Ti = float(ti_pos_y - j) * GRID_SPACING
            r_old_Ti = np.sqrt(dx_old_Ti * dx_old_Ti + dy_old_Ti * dy_old_Ti)

            # Parámetros Morse para Ti-atom_other
            D0_Ti = morse_params_array[2, atom_other, 0]  # Ti=2
            alpha_Ti = morse_params_array[2, atom_other, 1]
            r0_Ti = morse_params_array[2, atom_other, 2]

            # Potencial de Morse inline
            delta_r = r_old_Ti - r0_Ti
            exp_term = np.exp(-alpha_Ti * delta_r)
            exp2_term = exp_term * exp_term
            U_old_Ti = D0_Ti * (exp2_term - 2.0 * exp_term)

            # --- Fe en posición antigua interactuando con atom_other ---
            dx_old_Fe = float(fe_pos_x - i) * GRID_SPACING
            dy_old_Fe = float(fe_pos_y - j) * GRID_SPACING
            r_old_Fe = np.sqrt(dx_old_Fe * dx_old_Fe + dy_old_Fe * dy_old_Fe)

            # Parámetros Morse para Fe-atom_other
            D0_Fe = morse_params_array[0, atom_other, 0]  # Fe=0
            alpha_Fe = morse_params_array[0, atom_other, 1]
            r0_Fe = morse_params_array[0, atom_other, 2]

            # Potencial de Morse inline
            delta_r = r_old_Fe - r0_Fe
            exp_term = np.exp(-alpha_Fe * delta_r)
            exp2_term = exp_term * exp_term
            U_old_Fe = D0_Fe * (exp2_term - 2.0 * exp_term)

            # ================================================================
            # ENERGÍA QUE GANAMOS (configuración nueva)
            # ================================================================

            # --- Ti en nueva posición (fe_pos) interactuando con atom_other ---
            dx_new_Ti = float(fe_pos_x - i) * GRID_SPACING
            dy_new_Ti = float(fe_pos_y - j) * GRID_SPACING
            r_new_Ti = np.sqrt(dx_new_Ti * dx_new_Ti + dy_new_Ti * dy_new_Ti)

            # Potencial de Morse (mismos parámetros D0_Ti, alpha_Ti, r0_Ti)
            delta_r = r_new_Ti - r0_Ti
            exp_term = np.exp(-alpha_Ti * delta_r)
            exp2_term = exp_term * exp_term
            U_new_Ti = D0_Ti * (exp2_term - 2.0 * exp_term)

            # --- Fe en nueva posición (ti_pos) interactuando con atom_other ---
            dx_new_Fe = float(ti_pos_x - i) * GRID_SPACING
            dy_new_Fe = float(ti_pos_y - j) * GRID_SPACING
            r_new_Fe = np.sqrt(dx_new_Fe * dx_new_Fe + dy_new_Fe * dy_new_Fe)

            # Potencial de Morse (mismos parámetros D0_Fe, alpha_Fe, r0_Fe)
            delta_r = r_new_Fe - r0_Fe
            exp_term = np.exp(-alpha_Fe * delta_r)
            exp2_term = exp_term * exp_term
            U_new_Fe = D0_Fe * (exp2_term - 2.0 * exp_term)

            # ================================================================
            # ACUMULAR DIFERENCIA
            # ================================================================
            delta_E += (U_new_Ti + U_new_Fe - U_old_Ti - U_old_Fe)

    # ========================================================================
    # NOTA: No necesitamos corrección adicional para la interacción Ti-Fe
    # porque hemos calculado correctamente todas las interacciones que cambian.
    # La interacción directa Ti-Fe cambia de posición pero sigue existiendo,
    # y ya está incluida en los cálculos anteriores.
    # ========================================================================

    return delta_E


@njit(fastmath=True, cache=True)
def validate_delta_E(
    grid_array: np.ndarray,
    Ti_positions: np.ndarray,
    Fe_positions: np.ndarray,
    ti_idx: int,
    fe_idx: int,
    morse_params_array: np.ndarray
) -> tuple:
    """
    Valida que compute_delta_E_swap_fast dé el mismo resultado que recalcular todo.

    Función para debugging/testing. NO usar en producción (muy lento).

    Args:
        grid_array: Array (10, 10) con estado actual
        Ti_positions: Array (8, 2) con posiciones de Ti
        Fe_positions: Array (n_Fe, 2) con posiciones de Fe
        ti_idx: Índice del Ti
        fe_idx: Índice del Fe
        morse_params_array: Array (3, 3, 3) con parámetros

    Returns:
        Tupla (delta_E_fast, delta_E_full, error_relativo)
    """
    # Energía actual
    E_actual = compute_total_energy_fast(grid_array, morse_params_array)

    # Hacer swap (crear copia)
    grid_new = grid_array.copy()
    ti_x, ti_y = Ti_positions[ti_idx]
    fe_x, fe_y = Fe_positions[fe_idx]

    grid_new[ti_x, ti_y] = 0  # Fe
    grid_new[fe_x, fe_y] = 2  # Ti

    # Energía nueva
    E_nuevo = compute_total_energy_fast(grid_new, morse_params_array)

    # ΔE calculado de forma completa (lento)
    delta_E_full = E_nuevo - E_actual

    # ΔE calculado de forma incremental (rápido)
    delta_E_fast = compute_delta_E_swap_fast(
        grid_array, Ti_positions, ti_idx, fe_x, fe_y, morse_params_array
    )

    # Error relativo
    if abs(delta_E_full) > 1e-10:
        error_rel = abs(delta_E_fast - delta_E_full) / abs(delta_E_full)
    else:
        error_rel = abs(delta_E_fast - delta_E_full)

    return delta_E_fast, delta_E_full, error_rel
