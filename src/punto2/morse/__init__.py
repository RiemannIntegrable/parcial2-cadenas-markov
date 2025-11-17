"""
Módulo morse - Potencial de Morse optimizado con Numba.

Este módulo contiene implementaciones de alto rendimiento del potencial
de Morse para cálculos energéticos en el Problema 2.
"""

from .potential_numba import (
    morse_potential_fast,
    preparar_morse_params_array,
    MORSE_PARAMS_DICT
)

__all__ = [
    'morse_potential_fast',
    'preparar_morse_params_array',
    'MORSE_PARAMS_DICT'
]
