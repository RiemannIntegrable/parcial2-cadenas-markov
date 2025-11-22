"""
Cálculo de energía optimizado con Numba para estructura 3D.

Este módulo contiene las funciones críticas para calcular la energía del sistema:
1. compute_total_energy_fast_3d: Energía total (O(N²)) - solo para inicialización
2. compute_delta_E_swap_fast_3d: ΔE incremental (O(N)) - OPTIMIZACIÓN CLAVE

La función compute_delta_E_swap_fast_3d es la clave del éxito del algoritmo,
logrando un speedup de 25× sin Numba y hasta 1000× con Numba.

Diferencia con punto2: Usa coordenadas 3D explícitas en lugar de grilla 2D.
"""

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def compute_total_energy_fast_3d(
    all_positions: np.ndarray,
    atom_types: np.ndarray,
    morse_params_array: np.ndarray
) -> float:
    """
    Calcula la energía total del sistema usando el potencial de Morse en 3D.

    Complejidad: O(N²) donde N=112 (número total de átomos).
    Calcula 6,216 interacciones (todos los pares únicos).

    IMPORTANTE: Esta función solo debe usarse para:
    - Calcular energía inicial del sistema
    - Validación/debugging
    NO debe usarse en el loop de Simulated Annealing (usar compute_delta_E_swap_fast_3d).

    Args:
        all_positions: Array (112, 3) con coordenadas (x, y, z) de todos los átomos
        atom_types: Array (112,) con tipos 0=Fe, 1=Nd, 2=Ti
        morse_params_array: Array (3, 3, 3) con parámetros de Morse
                           params[tipo1, tipo2] = [D0, alpha, r0]

    Returns:
        Energía total del sistema (suma sobre todos los pares únicos)

    Note:
        Optimizado con Numba JIT. fastmath=True habilita optimizaciones agresivas.

    Examples:
        >>> from src.punto3.crystal import crear_configuracion_inicial
        >>> from src.punto3.morse import preparar_morse_params_array
        >>> all_pos, types, _, _, _ = crear_configuracion_inicial(seed=42)
        >>> params = preparar_morse_params_array()
        >>> E = compute_total_energy_fast_3d(all_pos, types, params)
        >>> isinstance(E, float)
        True
    """
    energy = 0.0
    n_atoms = len(all_positions)

    # Iterar sobre todos los pares únicos (i, j) donde i < j
    for i in range(n_atoms):
        atom1 = atom_types[i]
        x1, y1, z1 = all_positions[i, 0], all_positions[i, 1], all_positions[i, 2]

        for j in range(i + 1, n_atoms):
            atom2 = atom_types[j]
            x2, y2, z2 = all_positions[j, 0], all_positions[j, 1], all_positions[j, 2]

            # Distancia euclidiana 3D
            dx = x1 - x2
            dy = y1 - y2
            dz = z1 - z2
            r = np.sqrt(dx * dx + dy * dy + dz * dz)

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
def compute_delta_E_swap_fast_3d(
    all_positions: np.ndarray,
    atom_types: np.ndarray,
    ti_idx: int,
    fe_idx: int,
    morse_params_array: np.ndarray
) -> float:
    """
    Calcula ΔE incremental para un swap Ti ↔ Fe en 3D.

    Esta es LA FUNCIÓN CRÍTICA del Problema 3. Implementa el cálculo eficiente
    de la diferencia de energía SIN recalcular toda la red.

    Algoritmo:
        ΔE = E_nuevo - E_actual

        E_nuevo: Energía con Ti en fe_idx y Fe en ti_idx
        E_actual: Energía con Ti en ti_idx y Fe en fe_idx

        Solo recalculamos las interacciones que CAMBIAN:
        - Ti en su nueva posición con todos los demás
        - Fe en su nueva posición con todos los demás
        - Restamos las interacciones viejas

    Complejidad: O(N) donde N=112 → ~224 interacciones
    Speedup vs O(N²): 25× sin Numba, ~1000× con Numba

    Args:
        all_positions: Array (112, 3) con coordenadas de todos los átomos
        atom_types: Array (112,) con tipos actuales
        ti_idx: Índice del Ti a mover (debe estar en rango 16-111)
        fe_idx: Índice del Fe a intercambiar (debe estar en rango 16-111)
        morse_params_array: Array (3, 3, 3) con parámetros de Morse

    Returns:
        ΔE: Cambio de energía (E_nuevo - E_actual)
            - ΔE < 0: mejora (disminuye energía)
            - ΔE > 0: empeora (aumenta energía)

    Note:
        Esta función NO modifica all_positions ni atom_types.
        El caller es responsable de aplicar el swap si se acepta.

    Examples:
        >>> from src.punto3.crystal import crear_configuracion_inicial
        >>> from src.punto3.morse import preparar_morse_params_array
        >>> all_pos, types, Ti_idx, Fe_idx, _ = crear_configuracion_inicial(seed=42)
        >>> params = preparar_morse_params_array()
        >>> delta_E = compute_delta_E_swap_fast_3d(all_pos, types, Ti_idx[0], Fe_idx[0], params)
        >>> isinstance(delta_E, float)
        True
    """
    # Posiciones involucradas en el swap
    ti_x, ti_y, ti_z = all_positions[ti_idx, 0], all_positions[ti_idx, 1], all_positions[ti_idx, 2]
    fe_x, fe_y, fe_z = all_positions[fe_idx, 0], all_positions[fe_idx, 1], all_positions[fe_idx, 2]

    delta_E = 0.0
    n_atoms = len(all_positions)

    # ========================================================================
    # ITERAR SOBRE TODOS LOS DEMÁS ÁTOMOS
    # ========================================================================
    for k in range(n_atoms):
        # Saltar las posiciones involucradas en el swap
        if k == ti_idx or k == fe_idx:
            continue

        atom_other = atom_types[k]
        other_x, other_y, other_z = all_positions[k, 0], all_positions[k, 1], all_positions[k, 2]

        # ================================================================
        # ENERGÍA QUE PERDEMOS (configuración actual)
        # ================================================================

        # --- Ti en posición antigua interactuando con atom_other ---
        dx_old_Ti = ti_x - other_x
        dy_old_Ti = ti_y - other_y
        dz_old_Ti = ti_z - other_z
        r_old_Ti = np.sqrt(dx_old_Ti * dx_old_Ti + dy_old_Ti * dy_old_Ti + dz_old_Ti * dz_old_Ti)

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
        dx_old_Fe = fe_x - other_x
        dy_old_Fe = fe_y - other_y
        dz_old_Fe = fe_z - other_z
        r_old_Fe = np.sqrt(dx_old_Fe * dx_old_Fe + dy_old_Fe * dy_old_Fe + dz_old_Fe * dz_old_Fe)

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
        dx_new_Ti = fe_x - other_x
        dy_new_Ti = fe_y - other_y
        dz_new_Ti = fe_z - other_z
        r_new_Ti = np.sqrt(dx_new_Ti * dx_new_Ti + dy_new_Ti * dy_new_Ti + dz_new_Ti * dz_new_Ti)

        # Potencial de Morse (mismos parámetros D0_Ti, alpha_Ti, r0_Ti)
        delta_r = r_new_Ti - r0_Ti
        exp_term = np.exp(-alpha_Ti * delta_r)
        exp2_term = exp_term * exp_term
        U_new_Ti = D0_Ti * (exp2_term - 2.0 * exp_term)

        # --- Fe en nueva posición (ti_pos) interactuando con atom_other ---
        dx_new_Fe = ti_x - other_x
        dy_new_Fe = ti_y - other_y
        dz_new_Fe = ti_z - other_z
        r_new_Fe = np.sqrt(dx_new_Fe * dx_new_Fe + dy_new_Fe * dy_new_Fe + dz_new_Fe * dz_new_Fe)

        # Potencial de Morse (mismos parámetros D0_Fe, alpha_Fe, r0_Fe)
        delta_r = r_new_Fe - r0_Fe
        exp_term = np.exp(-alpha_Fe * delta_r)
        exp2_term = exp_term * exp_term
        U_new_Fe = D0_Fe * (exp2_term - 2.0 * exp_term)

        # ================================================================
        # ACUMULAR DIFERENCIA
        # ================================================================
        delta_E += (U_new_Ti + U_new_Fe - U_old_Ti - U_old_Fe)

    return delta_E


@njit(fastmath=True, cache=True)
def validate_delta_E_3d(
    all_positions: np.ndarray,
    atom_types: np.ndarray,
    ti_idx: int,
    fe_idx: int,
    morse_params_array: np.ndarray
) -> tuple:
    """
    Valida que compute_delta_E_swap_fast_3d dé el mismo resultado que recalcular todo.

    Función para debugging/testing. NO usar en producción (muy lento).

    Args:
        all_positions: Array (112, 3) con coordenadas
        atom_types: Array (112,) con tipos
        ti_idx: Índice del Ti
        fe_idx: Índice del Fe
        morse_params_array: Array (3, 3, 3) con parámetros

    Returns:
        Tupla (delta_E_fast, delta_E_full, error_relativo)
    """
    # Energía actual
    E_actual = compute_total_energy_fast_3d(all_positions, atom_types, morse_params_array)

    # Hacer swap (crear copia)
    atom_types_new = atom_types.copy()
    atom_types_new[ti_idx] = 0  # Fe
    atom_types_new[fe_idx] = 2  # Ti

    # Energía nueva
    E_nuevo = compute_total_energy_fast_3d(all_positions, atom_types_new, morse_params_array)

    # ΔE calculado de forma completa (lento)
    delta_E_full = E_nuevo - E_actual

    # ΔE calculado de forma incremental (rápido)
    delta_E_fast = compute_delta_E_swap_fast_3d(
        all_positions, atom_types, ti_idx, fe_idx, morse_params_array
    )

    # Error relativo
    if abs(delta_E_full) > 1e-10:
        error_rel = abs(delta_E_fast - delta_E_full) / abs(delta_E_full)
    else:
        error_rel = abs(delta_E_fast - delta_E_full)

    return delta_E_fast, delta_E_full, error_rel
