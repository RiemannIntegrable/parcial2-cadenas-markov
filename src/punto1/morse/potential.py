"""
Implementación del Potencial de Morse para interacciones atómicas.

El potencial de Morse es un modelo para la energía de interacción entre
dos átomos como función de su distancia. Tiene la forma:

    U(r) = D₀[exp(-2α(r - r₀)) - 2·exp(-α(r - r₀))]

donde:
    - r: distancia entre los átomos
    - D₀: profundidad del pozo de potencial
    - α: controla el ancho del pozo
    - r₀: distancia de equilibrio (mínimo de energía)
"""

import numpy as np
from typing import Union
from .parameters import get_morse_params


def morse_potential(
    r: Union[float, np.ndarray],
    D0: float,
    alpha: float,
    r0: float
) -> Union[float, np.ndarray]:
    """
    Calcula el potencial de Morse para una distancia o array de distancias.

    U(r) = D₀[exp(-2α(r - r₀)) - 2·exp(-α(r - r₀))]

    Args:
        r: Distancia entre átomos (escalar o array)
        D0: Profundidad del pozo de potencial
        alpha: Parámetro de ancho
        r0: Distancia de equilibrio

    Returns:
        Energía de interacción (mismo tipo que r)

    Examples:
        >>> # Energía en el equilibrio (debería ser -D₀)
        >>> U_eq = morse_potential(2.7361, D0=0.764, alpha=1.5995, r0=2.7361)
        >>> np.isclose(U_eq, -0.764)
        True

        >>> # Energía a distancia infinita (debería tender a 0)
        >>> U_inf = morse_potential(100.0, D0=0.764, alpha=1.5995, r0=2.7361)
        >>> np.isclose(U_inf, 0.0, atol=1e-6)
        True
    """
    delta_r = r - r0
    exp_term = np.exp(-alpha * delta_r)

    # U(r) = D₀[e^(-2α(r-r₀)) - 2e^(-α(r-r₀))]
    #      = D₀[(e^(-α(r-r₀)))² - 2e^(-α(r-r₀))]
    U = D0 * (exp_term**2 - 2 * exp_term)

    return U


def morse_potential_from_types(
    r: Union[float, np.ndarray],
    atom_type1: str,
    atom_type2: str
) -> Union[float, np.ndarray]:
    """
    Calcula el potencial de Morse usando los tipos de átomos.

    Esta función es un wrapper conveniente que busca automáticamente
    los parámetros correctos de la tabla.

    Args:
        r: Distancia entre átomos
        atom_type1: Tipo del primer átomo ('Fe', 'Nd', o 'Ti')
        atom_type2: Tipo del segundo átomo ('Fe', 'Nd', o 'Ti')

    Returns:
        Energía de interacción

    Examples:
        >>> U = morse_potential_from_types(2.914, 'Fe', 'Ti')
        >>> np.isclose(U, -0.8162)  # En equilibrio
        True
    """
    params = get_morse_params(atom_type1, atom_type2)
    return morse_potential(r, params['D0'], params['alpha'], params['r0'])


def morse_force(
    r: Union[float, np.ndarray],
    D0: float,
    alpha: float,
    r0: float
) -> Union[float, np.ndarray]:
    """
    Calcula la fuerza derivada del potencial de Morse.

    F(r) = -dU/dr = 2αD₀[exp(-2α(r-r₀)) - exp(-α(r-r₀))]

    Args:
        r: Distancia entre átomos
        D0: Profundidad del pozo
        alpha: Parámetro de ancho
        r0: Distancia de equilibrio

    Returns:
        Fuerza (negativa = atractiva, positiva = repulsiva)

    Note:
        Esta función es útil para simulaciones de dinámica molecular,
        pero no es necesaria para el problema de optimización actual.
    """
    delta_r = r - r0
    exp_term = np.exp(-alpha * delta_r)

    # F(r) = -dU/dr
    F = 2 * alpha * D0 * (exp_term**2 - exp_term)

    return F
