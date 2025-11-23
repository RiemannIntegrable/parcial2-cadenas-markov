"""
Punto 3: Optimización de Ti en Estructura Cristalina 3D NdFe₁₂

Este módulo implementa Simulated Annealing para encontrar la configuración óptima
de 8 átomos de Titanio (Ti) en una estructura cristalina 3D de NdFe₁₂.

Estructura:
- 16 átomos de Neodimio (Nd) - posiciones fijas
- 96 átomos de Hierro (Fe) - posiciones candidatas
- 8 átomos de Titanio (Ti) - sustituyen a 8 Fe (a optimizar)

Total: 112 átomos en 3D
"""

__version__ = "1.0.0"
__author__ = "José Miguel Acuña Hernández"
