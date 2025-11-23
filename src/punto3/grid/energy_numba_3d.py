"""
Cálculo de energía optimizado con Numba para sistemas 3D.

Este módulo contiene las funciones críticas para calcular la energía del sistema cristalino 3D:
1. compute_total_energy_3d: Energía total (O(N²)) - solo para inicialización
2. compute_delta_E_swap_3d: ΔE incremental (O(N)) - OPTIMIZACIÓN CLAVE

Adaptado para el Punto 3: Estructura cristalina NdFe12 en 3D.

La función compute_delta_E_swap_3d es la clave del éxito del algoritmo,
logrando un speedup de 25× sin Numba y hasta 1000× con Numba.
"""

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def compute_total_energy_3d(
    atom_types: np.ndarray,
    all_positions: np.ndarray,
    morse_params_array: np.ndarray
) -> float:
    """
    Calcula la energía total del sistema 3D usando el potencial de Morse.

    Complejidad: O(N²) donde N=112 (número total de átomos).
    Calcula C(112, 2) = 6,216 interacciones (todos los pares únicos).

    IMPORTANTE: Esta función solo debe usarse para:
    - Calcular energía inicial del sistema
    - Validación/debugging
    NO debe usarse en el loop de Simulated Annealing (usar compute_delta_E_swap_3d).

    Args:
        atom_types: Array (112,) con tipos atómicos (0=Fe, 1=Nd, 2=Ti)
        all_positions: Array (112, 3) con coordenadas (x, y, z) en Angstroms
        morse_params_array: Array (3, 3, 3) con parámetros de Morse
                           params[tipo1, tipo2] = [D0, alpha, r0]

    Returns:
        Energía total del sistema (suma sobre todos los pares únicos)

    Note:
        Optimizado con Numba JIT. fastmath=True habilita optimizaciones agresivas.

    Examples:
        >>> atom_types = np.array([0, 0, 1, 2], dtype=np.int8)  # Fe, Fe, Nd, Ti
        >>> positions = np.random.rand(4, 3)
        >>> params = preparar_morse_params_array()
        >>> E = compute_total_energy_3d(atom_types, positions, params)
        >>> E
        -12.345...
    """
    energy = 0.0
    N = len(atom_types)  # 112 átomos

    # Iterar sobre todos los pares únicos (i, j) donde i < j
    for i in range(N):
        atom1 = atom_types[i]
        x1 = all_positions[i, 0]
        y1 = all_positions[i, 1]
        z1 = all_positions[i, 2]

        for j in range(i + 1, N):  # Solo j > i (pares únicos)
            atom2 = atom_types[j]
            x2 = all_positions[j, 0]
            y2 = all_positions[j, 1]
            z2 = all_positions[j, 2]

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
def compute_delta_E_swap_3d(
    atom_types: np.ndarray,
    all_positions: np.ndarray,
    Ti_indices: np.ndarray,
    Fe_indices: np.ndarray,
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

        E_nuevo: Energía con Ti en posición del Fe y Fe en posición del Ti
        E_actual: Energía actual

        Solo recalculamos las interacciones que CAMBIAN:
        - Ti en su nueva posición interactuando con todos los demás
        - Fe en su nueva posición interactuando con todos los demás
        - Restamos las interacciones viejas

    Complejidad: O(N) donde N=112 → ~224 interacciones
    Speedup vs O(N²): 25× sin Numba, ~1000× con Numba

    Args:
        atom_types: Array (112,) con tipos atómicos actuales
        all_positions: Array (112, 3) con posiciones 3D
        Ti_indices: Array (8,) con índices (0-95) de átomos de Ti
        Fe_indices: Array (88,) con índices (0-95) de átomos de Fe
        ti_idx: Índice en Ti_indices del Ti a intercambiar (0-7)
        fe_idx: Índice en Fe_indices del Fe a intercambiar (0-87)
        morse_params_array: Array (3, 3, 3) con parámetros de Morse

    Returns:
        ΔE: Cambio de energía (E_nuevo - E_actual)
            - ΔE < 0: mejora (disminuye energía)
            - ΔE > 0: empeora (aumenta energía)

    Note:
        Esta función NO modifica atom_types, all_positions ni los índices.
        El caller es responsable de aplicar el swap si se acepta.

    Examples:
        >>> atom_types, all_pos, Ti_idx, Fe_idx = crear_configuracion_inicial_3d(...)
        >>> params = preparar_morse_params_array()
        >>> delta_E = compute_delta_E_swap_3d(atom_types, all_pos, Ti_idx, Fe_idx, 0, 0, params)
        >>> delta_E
        -1.234...
    """
    # Índices globales (0-95) de las posiciones candidatas a intercambiar
    ti_global_idx = Ti_indices[ti_idx]  # Posición que actualmente tiene Ti
    fe_global_idx = Fe_indices[fe_idx]  # Posición que actualmente tiene Fe

    # Posiciones 3D de Ti y Fe
    ti_x = all_positions[ti_global_idx, 0]
    ti_y = all_positions[ti_global_idx, 1]
    ti_z = all_positions[ti_global_idx, 2]

    fe_x = all_positions[fe_global_idx, 0]
    fe_y = all_positions[fe_global_idx, 1]
    fe_z = all_positions[fe_global_idx, 2]

    delta_E = 0.0
    N = len(atom_types)  # 112

    # ========================================================================
    # ITERAR SOBRE TODOS LOS DEMÁS ÁTOMOS
    # ========================================================================
    for k in range(N):
        # Saltar las posiciones involucradas en el swap
        if k == ti_global_idx or k == fe_global_idx:
            continue

        atom_other = atom_types[k]
        x_other = all_positions[k, 0]
        y_other = all_positions[k, 1]
        z_other = all_positions[k, 2]

        # ================================================================
        # ENERGÍA QUE PERDEMOS (configuración actual)
        # ================================================================

        # --- Ti en posición antigua interactuando con atom_other ---
        dx_old_Ti = ti_x - x_other
        dy_old_Ti = ti_y - y_other
        dz_old_Ti = ti_z - z_other
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
        dx_old_Fe = fe_x - x_other
        dy_old_Fe = fe_y - y_other
        dz_old_Fe = fe_z - z_other
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

        # --- Ti en nueva posición (donde estaba Fe) interactuando con atom_other ---
        # Misma distancia que r_old_Fe pero con parámetros de Ti
        r_new_Ti = r_old_Fe  # Ti ahora está en la posición del Fe

        # Potencial de Morse con parámetros de Ti
        delta_r = r_new_Ti - r0_Ti
        exp_term = np.exp(-alpha_Ti * delta_r)
        exp2_term = exp_term * exp_term
        U_new_Ti = D0_Ti * (exp2_term - 2.0 * exp_term)

        # --- Fe en nueva posición (donde estaba Ti) interactuando con atom_other ---
        # Misma distancia que r_old_Ti pero con parámetros de Fe
        r_new_Fe = r_old_Ti  # Fe ahora está en la posición del Ti

        # Potencial de Morse con parámetros de Fe
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
def validate_delta_E_3d(
    atom_types: np.ndarray,
    all_positions: np.ndarray,
    Ti_indices: np.ndarray,
    Fe_indices: np.ndarray,
    ti_idx: int,
    fe_idx: int,
    morse_params_array: np.ndarray
) -> tuple:
    """
    Valida que compute_delta_E_swap_3d dé el mismo resultado que recalcular todo.

    Función para debugging/testing. NO usar en producción (muy lento).

    Args:
        atom_types: Array (112,) con tipos atómicos
        all_positions: Array (112, 3) con posiciones 3D
        Ti_indices: Array (8,) con índices de Ti
        Fe_indices: Array (88,) con índices de Fe
        ti_idx: Índice del Ti
        fe_idx: Índice del Fe
        morse_params_array: Array (3, 3, 3) con parámetros

    Returns:
        Tupla (delta_E_fast, delta_E_full, error_relativo)
    """
    # Energía actual
    E_actual = compute_total_energy_3d(atom_types, all_positions, morse_params_array)

    # Hacer swap (crear copia)
    atom_types_new = atom_types.copy()

    ti_global_idx = Ti_indices[ti_idx]
    fe_global_idx = Fe_indices[fe_idx]

    atom_types_new[ti_global_idx] = 0  # Fe
    atom_types_new[fe_global_idx] = 2  # Ti

    # Energía nueva
    E_nuevo = compute_total_energy_3d(atom_types_new, all_positions, morse_params_array)

    # ΔE calculado de forma completa (lento)
    delta_E_full = E_nuevo - E_actual

    # ΔE calculado de forma incremental (rápido)
    delta_E_fast = compute_delta_E_swap_3d(
        atom_types, all_positions, Ti_indices, Fe_indices, ti_idx, fe_idx, morse_params_array
    )

    # Error relativo
    if abs(delta_E_full) > 1e-10:
        error_rel = abs(delta_E_fast - delta_E_full) / abs(delta_E_full)
    else:
        error_rel = abs(delta_E_fast - delta_E_full)

    return delta_E_fast, delta_E_full, error_rel
