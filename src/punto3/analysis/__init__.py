"""
Módulo analysis: Análisis espacial de configuraciones 3D

Provee funciones para analizar patrones espaciales en la estructura cristalina optimizada.
"""

from .spatial_analysis_3d import (
    calcular_distancias_Ti_Nd_3d,
    calcular_distancias_Ti_Ti_3d,
    calcular_centro_de_masa_3d,
    analizar_patron_espacial_3d,
    interpretar_patron_3d
)

__all__ = [
    'calcular_distancias_Ti_Nd_3d',
    'calcular_distancias_Ti_Ti_3d',
    'calcular_centro_de_masa_3d',
    'analizar_patron_espacial_3d',
    'interpretar_patron_3d'
]
