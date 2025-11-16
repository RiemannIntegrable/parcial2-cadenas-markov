"""
Módulo morse: Potencial de Morse y parámetros atómicos.

Este módulo implementa el potencial de Morse para calcular energías
de interacción entre átomos en estructuras cristalinas.

Exports principales:
    - morse_potential: Función principal del potencial de Morse
    - morse_potential_from_types: Wrapper conveniente usando tipos atómicos
    - get_morse_params: Obtener parámetros para un par de átomos
    - MORSE_PARAMETERS: Tabla completa de parámetros
    - ATOM_TYPES: Lista de tipos atómicos válidos
"""

from .potential import (
    morse_potential,
    morse_potential_from_types,
    morse_force
)

from .parameters import (
    MORSE_PARAMETERS,
    get_morse_params,
    ATOM_TYPES
)

__all__ = [
    'morse_potential',
    'morse_potential_from_types',
    'morse_force',
    'MORSE_PARAMETERS',
    'get_morse_params',
    'ATOM_TYPES'
]
