"""
Creación y manejo de grillas 2D para el Problema 1.

Este módulo define la estructura de la grilla 4×4 con átomos fijos
de tipo R (Neodimio) y Fe (Hierro), según la Figura 1 del problema.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class Grid2D:
    """
    Representa una grilla 2D con átomos de diferentes tipos.

    Attributes:
        R_positions: Array (n_R, 2) con coordenadas de átomos R (Neodimio)
        Fe_positions: Array (n_Fe, 2) con coordenadas de sitios Fe
        Ti_position_idx: Índice en Fe_positions donde está el átomo de Ti
                         (None si no hay Ti asignado)
        size: Tupla (filas, columnas) del tamaño de la grilla
    """
    R_positions: np.ndarray
    Fe_positions: np.ndarray
    Ti_position_idx: int = None
    size: Tuple[int, int] = field(default=(4, 4))

    def __post_init__(self):
        """Validación después de inicialización."""
        assert self.R_positions.shape[1] == 2, "R_positions debe ser (n, 2)"
        assert self.Fe_positions.shape[1] == 2, "Fe_positions debe ser (n, 2)"

        if self.Ti_position_idx is not None:
            n_Fe = len(self.Fe_positions)
            assert 0 <= self.Ti_position_idx < n_Fe, \
                f"Ti_position_idx debe estar en [0, {n_Fe})"

    @property
    def n_R(self) -> int:
        """Número de átomos de Neodimio."""
        return len(self.R_positions)

    @property
    def n_Fe_sites(self) -> int:
        """Número total de sitios para Fe/Ti."""
        return len(self.Fe_positions)

    def get_all_atoms(self) -> List[Tuple[np.ndarray, str]]:
        """
        Retorna lista de todos los átomos con sus posiciones y tipos.

        Returns:
            Lista de tuplas (posición, tipo) donde:
                - posición: np.ndarray de shape (2,)
                - tipo: 'Nd', 'Fe', o 'Ti'

        Examples:
            >>> grid = create_grid_4x4()
            >>> grid.Ti_position_idx = 0
            >>> atoms = grid.get_all_atoms()
            >>> len(atoms)
            16
        """
        atoms = []

        # Agregar átomos de Neodimio
        for pos in self.R_positions:
            atoms.append((pos, 'Nd'))

        # Agregar átomos Fe/Ti
        for i, pos in enumerate(self.Fe_positions):
            if i == self.Ti_position_idx:
                atoms.append((pos, 'Ti'))
            else:
                atoms.append((pos, 'Fe'))

        return atoms

    def set_Ti_position(self, idx: int) -> None:
        """
        Coloca el átomo de Ti en una posición específica.

        Args:
            idx: Índice en Fe_positions donde colocar el Ti

        Raises:
            AssertionError: Si idx está fuera de rango
        """
        assert 0 <= idx < self.n_Fe_sites, \
            f"Índice {idx} fuera de rango [0, {self.n_Fe_sites})"
        self.Ti_position_idx = idx

    def copy(self) -> 'Grid2D':
        """Crea una copia profunda de la grilla."""
        return Grid2D(
            R_positions=self.R_positions.copy(),
            Fe_positions=self.Fe_positions.copy(),
            Ti_position_idx=self.Ti_position_idx,
            size=self.size
        )


def create_grid_4x4() -> Grid2D:
    """
    Crea la grilla 2D de 4×4 según la Figura 1 del problema.

    La grilla tiene:
        - 4 átomos de R (Neodimio) en el centro: posiciones (2.8,2.8), (5.6,2.8), (2.8,5.6), (5.6,5.6) Å
        - 12 átomos de Fe en el resto de posiciones
        - Espaciado entre átomos adyacentes: 2.8 Å

    Returns:
        Objeto Grid2D inicializado

    Examples:
        >>> grid = create_grid_4x4()
        >>> grid.n_R
        4
        >>> grid.n_Fe_sites
        12
        >>> grid.size
        (4, 4)
    """
    # Constante de espaciado de la grilla (Angstroms)
    GRID_SPACING = 2.8

    # Posiciones de Neodimio (R) en el centro 2×2
    # Originalmente en (1,1), (2,1), (1,2), (2,2) → multiplicar por 2.8
    R_positions = np.array([
        [1.0 * GRID_SPACING, 1.0 * GRID_SPACING],
        [2.0 * GRID_SPACING, 1.0 * GRID_SPACING],
        [1.0 * GRID_SPACING, 2.0 * GRID_SPACING],
        [2.0 * GRID_SPACING, 2.0 * GRID_SPACING]
    ])

    # Posiciones de Fe: todas excepto las de R
    Fe_positions = []
    for x in range(4):
        for y in range(4):
            # Verificar si es una posición de R (comparar índices, no coordenadas)
            is_R = any(
                (x == int(r_pos[0] / GRID_SPACING) and y == int(r_pos[1] / GRID_SPACING))
                for r_pos in R_positions
            )
            if not is_R:
                # Convertir índices a coordenadas físicas (Angstroms)
                Fe_positions.append([float(x) * GRID_SPACING, float(y) * GRID_SPACING])

    Fe_positions = np.array(Fe_positions)

    grid = Grid2D(
        R_positions=R_positions,
        Fe_positions=Fe_positions,
        Ti_position_idx=None,
        size=(4, 4)
    )

    return grid


def get_Fe_positions_with_coords(grid: Grid2D) -> Dict[int, Tuple[float, float]]:
    """
    Retorna un diccionario mapeando índices de Fe a sus coordenadas (x, y) en Angstroms.

    Útil para análisis e interpretación de resultados.

    Args:
        grid: Grilla 2D

    Returns:
        Dict {índice: (x, y)} donde x, y son coordenadas en Angstroms

    Examples:
        >>> grid = create_grid_4x4()
        >>> coords = get_Fe_positions_with_coords(grid)
        >>> len(coords)
        12
    """
    return {
        i: (float(pos[0]), float(pos[1]))
        for i, pos in enumerate(grid.Fe_positions)
    }
