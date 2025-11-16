"""
Parámetros del Potencial de Morse para interacciones atómicas.

Este módulo contiene los parámetros experimentales de la Tabla 1 del problema,
que describen las interacciones entre pares de átomos en la estructura RT₁₂.

Referencias:
    Tabla 1 de parcial2.pdf - Parámetros basados en eV y Å
"""

from typing import Dict, Tuple

# Parámetros del Potencial de Morse: {(tipo1, tipo2): {'D0': ..., 'alpha': ..., 'r0': ...}}
# D0: Profundidad del pozo de potencial (energía)
# alpha: Parámetro de ancho del pozo (inverso de distancia)
# r0: Distancia de equilibrio (distancia)

MORSE_PARAMETERS: Dict[Tuple[str, str], Dict[str, float]] = {
    ('Fe', 'Fe'): {
        'D0': 0.764,
        'alpha': 1.5995,
        'r0': 2.7361
    },
    ('Fe', 'Nd'): {
        'D0': 0.6036,
        'alpha': 1.6458,
        'r0': 3.188
    },
    ('Nd', 'Nd'): {
        'D0': 0.312,
        'alpha': 0.945,
        'r0': 4.092
    },
    ('Fe', 'Ti'): {
        'D0': 0.8162,
        'alpha': 1.448,
        'r0': 2.914
    },
    ('Nd', 'Ti'): {
        'D0': 0.4964,
        'alpha': 1.4401,
        'r0': 3.4309
    },
    ('Ti', 'Ti'): {
        'D0': 0.6540,
        'alpha': 1.2118,
        'r0': 3.3476
    }
}


def get_morse_params(atom_type1: str, atom_type2: str) -> Dict[str, float]:
    """
    Obtiene los parámetros del potencial de Morse para un par de tipos atómicos.

    Los parámetros son simétricos, es decir, (Fe, Ti) = (Ti, Fe).

    Args:
        atom_type1: Tipo del primer átomo ('Fe', 'Nd', o 'Ti')
        atom_type2: Tipo del segundo átomo ('Fe', 'Nd', o 'Ti')

    Returns:
        Diccionario con las claves 'D0', 'alpha', 'r0'

    Raises:
        KeyError: Si el par de tipos no está en la tabla de parámetros

    Examples:
        >>> params = get_morse_params('Fe', 'Ti')
        >>> params['D0']
        0.8162
    """
    # Normalizar el orden (alfabético) para garantizar simetría
    pair = tuple(sorted([atom_type1, atom_type2]))

    if pair not in MORSE_PARAMETERS:
        raise KeyError(
            f"No hay parámetros de Morse para el par {pair}. "
            f"Pares válidos: {list(MORSE_PARAMETERS.keys())}"
        )

    return MORSE_PARAMETERS[pair]


# Constantes útiles
ATOM_TYPES = ['Fe', 'Nd', 'Ti']
"""Tipos de átomos válidos en el sistema"""
