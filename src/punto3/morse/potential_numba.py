"""
Potencial de Morse optimizado con Numba para máximo rendimiento (versión 3D).

Este módulo implementa el potencial de Morse usando compilación JIT con Numba,
logrando speedups de 100-1000× en funciones críticas.

La única diferencia con punto2 es que usamos distancia_3d en lugar de distancia_2d.

Convención de índices:
    0 = Fe (Hierro)
    1 = Nd (Neodimio)
    2 = Ti (Titanio)
"""

import numpy as np
from numba import njit
from typing import Dict, Tuple

# ============================================================================
# PARÁMETROS DEL POTENCIAL DE MORSE (Tabla 1 del problema)
# ============================================================================

MORSE_PARAMS_DICT: Dict[Tuple[str, str], Tuple[float, float, float]] = {
    ('Fe', 'Fe'): (0.764, 1.5995, 2.7361),
    ('Fe', 'Nd'): (0.6036, 1.6458, 3.188),
    ('Fe', 'Ti'): (0.8162, 1.448, 2.914),
    ('Nd', 'Nd'): (0.312, 0.945, 4.092),
    ('Nd', 'Ti'): (0.4964, 1.4401, 3.4309),
    ('Ti', 'Ti'): (0.6540, 1.2118, 3.3476)
}
"""Parámetros de Morse (D0, alpha, r0) para cada par de tipos atómicos."""


def preparar_morse_params_array() -> np.ndarray:
    """
    Prepara los parámetros de Morse en formato de array 3D para Numba.

    Crea un array de shape (3, 3, 3) donde:
        params[i, j, :] = [D0, alpha, r0] para interacción tipo_i <-> tipo_j

    Índices:
        0 = Fe, 1 = Nd, 2 = Ti

    Returns:
        np.ndarray: Array (3, 3, 3) dtype=float64 con parámetros

    Examples:
        >>> params = preparar_morse_params_array()
        >>> D0_Fe_Ti, alpha_Fe_Ti, r0_Fe_Ti = params[0, 2]
        >>> D0_Fe_Ti
        0.8162
    """
    # Mapeo de nombres a índices
    type_to_idx = {'Fe': 0, 'Nd': 1, 'Ti': 2}

    # Crear array 3D: [tipo1, tipo2, parámetro]
    params = np.zeros((3, 3, 3), dtype=np.float64)

    # Llenar con parámetros (simétrico)
    for (type1, type2), (D0, alpha, r0) in MORSE_PARAMS_DICT.items():
        idx1 = type_to_idx[type1]
        idx2 = type_to_idx[type2]

        params[idx1, idx2, 0] = D0
        params[idx1, idx2, 1] = alpha
        params[idx1, idx2, 2] = r0

        # Simetría
        params[idx2, idx1, 0] = D0
        params[idx2, idx1, 1] = alpha
        params[idx2, idx1, 2] = r0

    return params


# ============================================================================
# FUNCIONES OPTIMIZADAS CON NUMBA (VERSION 3D)
# ============================================================================

@njit(fastmath=True, cache=True, inline='always')
def morse_potential_fast(r: float, D0: float, alpha: float, r0: float) -> float:
    """
    Calcula el potencial de Morse entre dos átomos.

    Optimizado con Numba JIT compilation para máximo rendimiento.

    Fórmula:
        U(r) = D₀[exp(-2α(r-r₀)) - 2·exp(-α(r-r₀))]

    Args:
        r: Distancia entre átomos
        D0: Profundidad del pozo de potencial
        alpha: Parámetro de ancho del pozo
        r0: Distancia de equilibrio

    Returns:
        Energía del potencial de Morse

    Note:
        Esta función es inlined por el compilador Numba para máximo rendimiento.
        fastmath=True habilita optimizaciones matemáticas agresivas.
    """
    delta_r = r - r0
    exp_term = np.exp(-alpha * delta_r)
    exp2_term = exp_term * exp_term  # Más rápido que exp(-2*alpha*delta_r)

    return D0 * (exp2_term - 2.0 * exp_term)


@njit(fastmath=True, cache=True, inline='always')
def distancia_3d(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    """
    Calcula la distancia euclidiana 3D entre dos puntos.

    Optimizado con Numba para máximo rendimiento.

    Args:
        x1, y1, z1: Coordenadas del primer punto
        x2, y2, z2: Coordenadas del segundo punto

    Returns:
        Distancia euclidiana

    Examples:
        >>> d = distancia_3d(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        >>> np.isclose(d, np.sqrt(3))
        True
    """
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    return np.sqrt(dx * dx + dy * dy + dz * dz)


@njit(fastmath=True, cache=True, inline='always')
def distancia_3d_cuadrada(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    """
    Calcula la distancia euclidiana al cuadrado (más rápido, evita sqrt).

    Útil cuando solo se necesita comparar distancias.

    Args:
        x1, y1, z1: Coordenadas del primer punto
        x2, y2, z2: Coordenadas del segundo punto

    Returns:
        Distancia euclidiana al cuadrado
    """
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    return dx * dx + dy * dy + dz * dz


# ============================================================================
# UTILIDADES
# ============================================================================

def get_morse_params_by_names(atom_type1: str, atom_type2: str) -> Tuple[float, float, float]:
    """
    Obtiene parámetros de Morse para un par de tipos atómicos (por nombre).

    Args:
        atom_type1: Tipo del primer átomo ('Fe', 'Nd', o 'Ti')
        atom_type2: Tipo del segundo átomo ('Fe', 'Nd', o 'Ti')

    Returns:
        Tupla (D0, alpha, r0)

    Raises:
        KeyError: Si el par no existe

    Examples:
        >>> D0, alpha, r0 = get_morse_params_by_names('Fe', 'Ti')
        >>> D0
        0.8162
    """
    # Normalizar orden (alfabético)
    pair = tuple(sorted([atom_type1, atom_type2]))

    if pair not in MORSE_PARAMS_DICT:
        raise KeyError(
            f"No hay parámetros de Morse para el par {pair}. "
            f"Pares válidos: {list(MORSE_PARAMS_DICT.keys())}"
        )

    return MORSE_PARAMS_DICT[pair]


# Crear array global de parámetros para uso rápido
_MORSE_PARAMS_ARRAY_CACHE = None


def get_morse_params_array() -> np.ndarray:
    """
    Obtiene el array 3D de parámetros (con caché).

    Returns:
        Array (3, 3, 3) con parámetros de Morse
    """
    global _MORSE_PARAMS_ARRAY_CACHE

    if _MORSE_PARAMS_ARRAY_CACHE is None:
        _MORSE_PARAMS_ARRAY_CACHE = preparar_morse_params_array()

    return _MORSE_PARAMS_ARRAY_CACHE
