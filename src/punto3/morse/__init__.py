"""
Módulo morse: Potencial de Morse para interacciones atómicas

Provee funciones para calcular el potencial de Morse y preparar
los parámetros para todos los pares de tipos atómicos.
"""

from .potential_numba import (
    morse_potential,
    preparar_morse_params_array
)

__all__ = [
    'morse_potential',
    'preparar_morse_params_array'
]
