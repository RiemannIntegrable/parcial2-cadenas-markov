"""
Utilidades para crear y manipular la grilla 10×10.

Este módulo usa arrays numpy puros para compatibilidad con Numba.

Convención de valores en grid_array:
    0 = Fe (Hierro) - puede moverse
    1 = Nd (Neodimio) - FIJO en núcleo 4×4 central
    2 = Ti (Titanio) - puede moverse (8 átomos)
"""

import numpy as np
from typing import Tuple, Optional


def get_Nd_positions_fijas() -> np.ndarray:
    """
    Retorna las posiciones fijas del núcleo de Nd (4×4 central).

    El núcleo está en posiciones (x,y) donde x,y ∈ {3, 4, 5, 6}.

    Returns:
        np.ndarray: Array de shape (16, 2) con coordenadas (x, y) de los 16 Nd

    Examples:
        >>> Nd_pos = get_Nd_positions_fijas()
        >>> Nd_pos.shape
        (16, 2)
        >>> Nd_pos[0]
        array([3, 3])
    """
    positions = []
    for x in [3, 4, 5, 6]:
        for y in [3, 4, 5, 6]:
            positions.append([x, y])

    return np.array(positions, dtype=np.int8)


def crear_grid_inicial(seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crea la configuración inicial del sistema con 8 Ti aleatorios.

    Configuración:
        - Grilla 10×10
        - 16 Nd fijos en núcleo 4×4 central (x,y ∈ {3,4,5,6})
        - 8 Ti en posiciones aleatorias (fuera del núcleo)
        - 76 Fe en el resto

    Args:
        seed: Semilla para reproducibilidad (opcional)

    Returns:
        Tupla (grid_array, Ti_positions, Nd_positions):
            - grid_array: np.ndarray shape (10, 10), dtype=int8
                          Valores: 0=Fe, 1=Nd, 2=Ti
            - Ti_positions: np.ndarray shape (8, 2), dtype=int8
                           Coordenadas (x, y) de los 8 átomos de Ti
            - Nd_positions: np.ndarray shape (16, 2), dtype=int8
                           Coordenadas (x, y) de los 16 átomos de Nd (constante)

    Examples:
        >>> grid, Ti_pos, Nd_pos = crear_grid_inicial(seed=42)
        >>> grid.shape
        (10, 10)
        >>> Ti_pos.shape
        (8, 2)
        >>> np.sum(grid == 2)  # Contar Ti
        8
    """
    if seed is not None:
        np.random.seed(seed)

    # Inicializar grilla con Fe (0)
    grid_array = np.zeros((10, 10), dtype=np.int8)

    # Posiciones fijas de Nd (núcleo 4×4 central)
    Nd_positions = get_Nd_positions_fijas()

    # Colocar Nd en la grilla
    for x, y in Nd_positions:
        grid_array[x, y] = 1  # Nd = 1

    # Obtener posiciones disponibles (donde hay Fe, fuera del núcleo)
    posiciones_disponibles = []
    for x in range(10):
        for y in range(10):
            if grid_array[x, y] == 0:  # Solo Fe
                posiciones_disponibles.append([x, y])

    # Deben haber 84 posiciones disponibles (100 - 16 Nd = 84)
    assert len(posiciones_disponibles) == 84, \
        f"Esperadas 84 posiciones disponibles, encontradas {len(posiciones_disponibles)}"

    # Elegir 8 posiciones aleatorias para Ti
    indices_Ti = np.random.choice(len(posiciones_disponibles), size=8, replace=False)
    Ti_positions = np.array([posiciones_disponibles[i] for i in indices_Ti], dtype=np.int8)

    # Colocar Ti en la grilla
    for x, y in Ti_positions:
        grid_array[x, y] = 2  # Ti = 2

    # Verificaciones
    assert np.sum(grid_array == 0) == 76, f"Esperados 76 Fe, encontrados {np.sum(grid_array == 0)}"
    assert np.sum(grid_array == 1) == 16, f"Esperados 16 Nd, encontrados {np.sum(grid_array == 1)}"
    assert np.sum(grid_array == 2) == 8, f"Esperados 8 Ti, encontrados {np.sum(grid_array == 2)}"

    return grid_array, Ti_positions, Nd_positions


def get_Ti_positions(grid_array: np.ndarray) -> np.ndarray:
    """
    Extrae las posiciones de los átomos de Ti de la grilla.

    Args:
        grid_array: Array (10, 10) con valores 0=Fe, 1=Nd, 2=Ti

    Returns:
        np.ndarray: Array shape (n_Ti, 2) con coordenadas (x, y) de Ti

    Examples:
        >>> grid = np.zeros((10, 10), dtype=np.int8)
        >>> grid[2, 5] = 2  # Ti en (2, 5)
        >>> grid[7, 3] = 2  # Ti en (7, 3)
        >>> Ti_pos = get_Ti_positions(grid)
        >>> Ti_pos.shape
        (2, 2)
    """
    return np.argwhere(grid_array == 2).astype(np.int8)


def get_Fe_positions(grid_array: np.ndarray) -> np.ndarray:
    """
    Extrae las posiciones de los átomos de Fe de la grilla.

    Args:
        grid_array: Array (10, 10) con valores 0=Fe, 1=Nd, 2=Ti

    Returns:
        np.ndarray: Array shape (n_Fe, 2) con coordenadas (x, y) de Fe

    Examples:
        >>> grid = np.ones((10, 10), dtype=np.int8)  # Todo Nd
        >>> grid[0, 0] = 0  # Fe en (0, 0)
        >>> Fe_pos = get_Fe_positions(grid)
        >>> Fe_pos.shape
        (1, 2)
    """
    return np.argwhere(grid_array == 0).astype(np.int8)


def aplicar_swap(
    grid_array: np.ndarray,
    Ti_positions: np.ndarray,
    Fe_positions: np.ndarray,
    ti_idx: int,
    fe_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aplica un swap entre un Ti y un Fe.

    Esta función NO modifica los arrays originales, retorna copias.

    Args:
        grid_array: Array (10, 10) actual
        Ti_positions: Array (8, 2) con posiciones de Ti
        Fe_positions: Array (n_Fe, 2) con posiciones de Fe
        ti_idx: Índice del Ti a intercambiar (0-7)
        fe_idx: Índice del Fe a intercambiar

    Returns:
        Tupla (grid_new, Ti_positions_new, Fe_positions_new) con el swap aplicado

    Note:
        Esta función crea copias para evitar modificar el estado original.
        Para modificaciones in-place (más eficiente), usar directamente en el loop de SA.

    Examples:
        >>> grid = np.zeros((10, 10), dtype=np.int8)
        >>> grid[2, 5] = 2  # Ti
        >>> grid[7, 3] = 0  # Fe
        >>> Ti_pos = np.array([[2, 5]], dtype=np.int8)
        >>> Fe_pos = np.array([[7, 3]], dtype=np.int8)
        >>> grid_new, Ti_new, Fe_new = aplicar_swap(grid, Ti_pos, Fe_pos, 0, 0)
        >>> grid_new[2, 5]  # Ahora es Fe
        0
        >>> grid_new[7, 3]  # Ahora es Ti
        2
    """
    # Copiar arrays
    grid_new = grid_array.copy()
    Ti_new = Ti_positions.copy()
    Fe_new = Fe_positions.copy()

    # Obtener posiciones
    ti_x, ti_y = Ti_positions[ti_idx]
    fe_x, fe_y = Fe_positions[fe_idx]

    # Swap en la grilla
    grid_new[ti_x, ti_y] = 0  # Posición del Ti ahora tiene Fe
    grid_new[fe_x, fe_y] = 2  # Posición del Fe ahora tiene Ti

    # Actualizar arrays de posiciones
    Ti_new[ti_idx, 0] = fe_x
    Ti_new[ti_idx, 1] = fe_y

    Fe_new[fe_idx, 0] = ti_x
    Fe_new[fe_idx, 1] = ti_y

    return grid_new, Ti_new, Fe_new


def grid_to_string(grid_array: np.ndarray) -> str:
    """
    Convierte la grilla a una representación en string para visualización.

    Args:
        grid_array: Array (10, 10) con valores 0=Fe, 1=Nd, 2=Ti

    Returns:
        String con representación visual de la grilla

    Examples:
        >>> grid = np.zeros((10, 10), dtype=np.int8)
        >>> grid[3:7, 3:7] = 1  # Núcleo Nd
        >>> grid[0, 0] = 2      # Ti
        >>> print(grid_to_string(grid))
        T . . . . . . . . .
        . . . . . . . . . .
        . . . . . . . . . .
        . . . N N N N . . .
        ...
    """
    symbols = {0: '.', 1: 'N', 2: 'T'}
    lines = []
    for i in range(10):
        line = ' '.join(symbols[grid_array[i, j]] for j in range(10))
        lines.append(line)
    return '\n'.join(lines)
