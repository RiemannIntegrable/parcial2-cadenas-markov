"""
Problema 3: Optimización de Ti en Estructura Cristalina 3D (NdFe12)

Este módulo implementa Simulated Annealing para encontrar la configuración
óptima de 8 átomos de Ti sustituyendo a 8 átomos de Fe en la estructura
cristalina real de NdFe12 (supercelda 2×2×1).

Diferencias con Punto 2:
- Coordenadas 3D reales (Å) en lugar de grilla 2D
- 96 posiciones candidatas de Fe (vs 76 en grilla 10×10)
- Distancias euclidianas 3D para el potencial de Morse

Estructura de módulos:
- morse: Potencial de Morse optimizado con Numba (idéntico a punto2)
- crystal: Utilidades para manejar coordenadas 3D y posiciones atómicas
- optimization: Simulated Annealing adaptado a 3D
- analysis: Análisis espacial de patrones en 3D
- visualization: Gráficas 3D de configuraciones
"""

from .morse import preparar_morse_params_array
from .crystal import (
    get_Nd_positions_fijas,
    get_Fe_positions_all,
    crear_configuracion_inicial
)

__all__ = [
    'preparar_morse_params_array',
    'get_Nd_positions_fijas',
    'get_Fe_positions_all',
    'crear_configuracion_inicial'
]
