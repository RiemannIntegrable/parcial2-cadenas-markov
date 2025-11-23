"""
Cálculo de energía total del sistema cristalino 3D

Este módulo provee funciones optimizadas con Numba para calcular la energía
total del sistema usando el potencial de Morse entre todos los pares de átomos.
"""

import numpy as np
from numba import njit


@njit(fastmath=True, inline='always')
def morse_potential_inline(r: float, D0: float, alpha: float, r0: float) -> float:
    """
    Calcula el potencial de Morse entre dos átomos.

    U(r) = D₀[exp(-2α(r-r₀)) - 2exp(-α(r-r₀))]

    Args:
        r: Distancia entre átomos (Angstroms)
        D0: Profundidad del pozo (eV)
        alpha: Parámetro de alcance (1/Å)
        r0: Distancia de equilibrio (Å)

    Returns:
        Energía potencial (eV)
    """
    delta_r = r - r0
    exp_term = np.exp(-alpha * delta_r)
    exp2_term = exp_term * exp_term

    return D0 * (exp2_term - 2.0 * exp_term)


@njit(fastmath=True, cache=True)
def compute_total_energy_fast_3d(
    all_positions: np.ndarray,
    atom_types: np.ndarray,
    morse_params_array: np.ndarray
) -> float:
    """
    Calcula la energía total del sistema cristalino 3D.

    Suma el potencial de Morse sobre todos los pares únicos de átomos (i, j) con i < j.
    Optimizado con Numba JIT para máxima eficiencia.

    Args:
        all_positions: Array (N, 3) con coordenadas (x, y, z) de todos los átomos
        atom_types: Array (N,) con tipos [0=Fe, 1=Nd, 2=Ti]
        morse_params_array: Array (3, 3, 3) con parámetros Morse
                            Shape: (tipo1, tipo2, [D0, alpha, r0])

    Returns:
        Energía total del sistema (eV)

    Note:
        La complejidad es O(N²) donde N=112. Para un sistema tan pequeño,
        el cálculo completo es aceptablemente rápido incluso sin optimizaciones
        como listas de vecinos.
    """
    N = len(all_positions)
    energia_total = 0.0

    # Loop sobre todos los pares únicos (i, j) con i < j
    for i in range(N):
        tipo_i = atom_types[i]
        x_i, y_i, z_i = all_positions[i, 0], all_positions[i, 1], all_positions[i, 2]

        for j in range(i + 1, N):
            tipo_j = atom_types[j]
            x_j, y_j, z_j = all_positions[j, 0], all_positions[j, 1], all_positions[j, 2]

            # Calcular distancia euclidiana 3D
            dx = x_i - x_j
            dy = y_i - y_j
            dz = z_i - z_j
            r = np.sqrt(dx*dx + dy*dy + dz*dz)

            # Obtener parámetros de Morse para este par de tipos
            D0 = morse_params_array[tipo_i, tipo_j, 0]
            alpha = morse_params_array[tipo_i, tipo_j, 1]
            r0 = morse_params_array[tipo_i, tipo_j, 2]

            # Acumular energía
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
    Calcula la contribución a la energía total de UN átomo específico.

    Suma el potencial de Morse entre el átomo 'atom_idx' y TODOS los demás átomos.
    Esta función es CRÍTICA para el cálculo incremental de ΔE en Simulated Annealing.

    Args:
        atom_idx: Índice global del átomo
        all_positions: Array (N, 3) con todas las posiciones
        atom_types: Array (N,) con tipos de átomos
        morse_params_array: Array (3, 3, 3) con parámetros Morse

    Returns:
        Suma de energías de interacción del átomo con todos los demás

    Note:
        Esta energía se cuenta UNA VEZ por átomo, no dos veces como en la energía total.
        Para calcular ΔE en un swap:
            ΔE = (E_after_i + E_after_j) - (E_before_i + E_before_j)
    """
    N = len(all_positions)
    tipo_atom = atom_types[atom_idx]
    x_atom = all_positions[atom_idx, 0]
    y_atom = all_positions[atom_idx, 1]
    z_atom = all_positions[atom_idx, 2]

    energia_contribucion = 0.0

    for other_idx in range(N):
        if other_idx == atom_idx:
            continue  # No interactuar consigo mismo

        tipo_other = atom_types[other_idx]
        x_other = all_positions[other_idx, 0]
        y_other = all_positions[other_idx, 1]
        z_other = all_positions[other_idx, 2]

        # Distancia euclidiana 3D
        dx = x_atom - x_other
        dy = y_atom - y_other
        dz = z_atom - z_other
        r = np.sqrt(dx*dx + dy*dy + dz*dz)

        # Parámetros de Morse
        D0 = morse_params_array[tipo_atom, tipo_other, 0]
        alpha = morse_params_array[tipo_atom, tipo_other, 1]
        r0 = morse_params_array[tipo_atom, tipo_other, 2]

        # Acumular energía
        energia_contribucion += morse_potential_inline(r, D0, alpha, r0)

    return energia_contribucion
