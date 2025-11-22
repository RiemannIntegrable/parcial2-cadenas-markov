"""Potencial de Morse optimizado con Numba."""

from .potential_numba import (
    preparar_morse_params_array,
    morse_potential_fast,
    distancia_3d,
    distancia_3d_cuadrada,
    get_morse_params_by_names,
    get_morse_params_array
)

__all__ = [
    'preparar_morse_params_array',
    'morse_potential_fast',
    'distancia_3d',
    'distancia_3d_cuadrada',
    'get_morse_params_by_names',
    'get_morse_params_array'
]
