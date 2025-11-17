"""
Módulo analysis - Análisis espacial de configuraciones óptimas.

Contiene funciones optimizadas con Numba para analizar patrones espaciales
de la distribución de Ti en la grilla, necesarias para el Punto 3.
"""

from .spatial_analysis_numba import (
    calcular_distancias_Ti_Nd,
    calcular_distancias_Ti_Ti,
    calcular_clustering_score,
    analizar_patron_espacial,
    analizar_distribucion_radial,
    interpretar_patron
)

__all__ = [
    'calcular_distancias_Ti_Nd',
    'calcular_distancias_Ti_Ti',
    'calcular_clustering_score',
    'analizar_patron_espacial',
    'analizar_distribucion_radial',
    'interpretar_patron'
]
