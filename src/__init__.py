"""
Paquete principal para optimización de configuraciones atómicas.

Este paquete implementa herramientas para resolver problemas de optimización
combinatoria en diseño de materiales usando Simulated Annealing y otros
métodos basados en Cadenas de Markov de Monte Carlo (MCMC).

Estructura:
    - punto1: Problema 1 (Grilla 4x4 con 1 átomo de Ti)
    - punto2: Problema 2 (Grilla 10x10 con 8 átomos de Ti) [pendiente]
    - punto3: Problema 3 (Estructura 3D NdFe₁₄) [pendiente]

Ejemplos de Uso:
    >>> # Importar directamente desde submódulo punto1
    >>> from src.punto1 import create_grid_4x4, brute_force_search
    >>> from src.punto1 import simulated_annealing, plot_grid_configuration

    >>> # Crear grilla y resolver
    >>> grid = create_grid_4x4()
    >>> result = brute_force_search(grid)
    >>> print(f"Óptimo: {result['best_position_idx']}, E = {result['best_energy']:.4f}")
"""

__version__ = '1.0.0'

# Los submódulos deben importarse explícitamente
# from src.punto1 import ...
# from src.punto2 import ...
# from src.punto3 import ...

__all__ = ['punto1']
