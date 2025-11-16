"""
Cálculo de energía total para configuraciones de grillas 2D.

Este módulo implementa el cálculo de la energía cohesiva total usando
el potencial de Morse para todos los pares de átomos en la grilla.
"""

import numpy as np
from typing import List, Tuple
from .grid_2d import Grid2D
from ..morse import morse_potential_from_types


def compute_distance_2d(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    Calcula la distancia euclidiana 2D entre dos posiciones.

    Args:
        pos1: Posición (x, y) del primer átomo
        pos2: Posición (x, y) del segundo átomo

    Returns:
        Distancia euclidiana

    Examples:
        >>> d = compute_distance_2d(np.array([0, 0]), np.array([3, 4]))
        >>> np.isclose(d, 5.0)
        True
    """
    return np.linalg.norm(pos1 - pos2)


def compute_total_energy(grid: Grid2D) -> float:
    """
    Calcula la energía cohesiva total de una configuración.

    La energía total se calcula como:
        E_total = Σᵢ Σⱼ>ᵢ U(rᵢⱼ)

    donde U es el potencial de Morse entre los átomos i y j,
    y rᵢⱼ es la distancia entre ellos.

    Args:
        grid: Grilla 2D con configuración de átomos

    Returns:
        Energía total del sistema

    Raises:
        ValueError: Si grid.Ti_position_idx es None (no hay Ti asignado)

    Examples:
        >>> grid = create_grid_4x4()
        >>> grid.set_Ti_position(0)
        >>> E = compute_total_energy(grid)
        >>> isinstance(E, float)
        True

    Note:
        La complejidad es O(N²) donde N es el número total de átomos.
        Para grilla 4×4, N = 16, así que esto es muy eficiente.
    """
    if grid.Ti_position_idx is None:
        raise ValueError(
            "No se ha asignado posición de Ti. "
            "Use grid.set_Ti_position(idx) primero."
        )

    # Obtener lista de todos los átomos con tipos
    atoms = grid.get_all_atoms()
    n_atoms = len(atoms)

    total_energy = 0.0

    # Sumar sobre todos los pares únicos (i, j) con j > i
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            pos_i, type_i = atoms[i]
            pos_j, type_j = atoms[j]

            # Calcular distancia
            r_ij = compute_distance_2d(pos_i, pos_j)

            # Calcular energía de interacción usando potencial de Morse
            U_ij = morse_potential_from_types(r_ij, type_i, type_j)

            # Acumular energía total
            total_energy += U_ij

    return total_energy


def compute_pairwise_energies(grid: Grid2D) -> List[Tuple[int, int, float, float]]:
    """
    Calcula las energías de interacción de todos los pares.

    Útil para análisis detallado de qué interacciones dominan la energía.

    Args:
        grid: Grilla 2D

    Returns:
        Lista de tuplas (i, j, distancia, energía) para cada par (i, j)

    Examples:
        >>> grid = create_grid_4x4()
        >>> grid.set_Ti_position(0)
        >>> pairs = compute_pairwise_energies(grid)
        >>> len(pairs) == 16 * 15 // 2  # n(n-1)/2 pares
        True
    """
    if grid.Ti_position_idx is None:
        raise ValueError("No se ha asignado posición de Ti.")

    atoms = grid.get_all_atoms()
    n_atoms = len(atoms)

    pairwise = []

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            pos_i, type_i = atoms[i]
            pos_j, type_j = atoms[j]

            r_ij = compute_distance_2d(pos_i, pos_j)
            U_ij = morse_potential_from_types(r_ij, type_i, type_j)

            pairwise.append((i, j, r_ij, U_ij))

    return pairwise


def compute_Ti_contributions(grid: Grid2D) -> dict:
    """
    Analiza las contribuciones energéticas del átomo de Ti.

    Calcula la energía de interacción del Ti con:
        - Átomos de Nd
        - Átomos de Fe

    Args:
        grid: Grilla 2D

    Returns:
        Diccionario con:
            - 'Ti_Nd_energy': Energía total Ti-Nd
            - 'Ti_Fe_energy': Energía total Ti-Fe
            - 'Ti_Nd_distances': Lista de distancias Ti-Nd
            - 'Ti_Fe_distances': Lista de distancias Ti-Fe
            - 'closest_Nd_distance': Distancia al Nd más cercano
            - 'closest_Fe_distance': Distancia al Fe más cercano

    Examples:
        >>> grid = create_grid_4x4()
        >>> grid.set_Ti_position(0)
        >>> contrib = compute_Ti_contributions(grid)
        >>> 'Ti_Nd_energy' in contrib
        True
    """
    if grid.Ti_position_idx is None:
        raise ValueError("No se ha asignado posición de Ti.")

    atoms = grid.get_all_atoms()
    Ti_position = grid.Fe_positions[grid.Ti_position_idx]

    Ti_Nd_energy = 0.0
    Ti_Fe_energy = 0.0
    Ti_Nd_distances = []
    Ti_Fe_distances = []

    for pos, atom_type in atoms:
        # Skip el propio Ti
        if np.array_equal(pos, Ti_position):
            continue

        r = compute_distance_2d(Ti_position, pos)

        if atom_type == 'Nd':
            U = morse_potential_from_types(r, 'Ti', 'Nd')
            Ti_Nd_energy += U
            Ti_Nd_distances.append(r)

        elif atom_type == 'Fe':
            U = morse_potential_from_types(r, 'Ti', 'Fe')
            Ti_Fe_energy += U
            Ti_Fe_distances.append(r)

    return {
        'Ti_Nd_energy': Ti_Nd_energy,
        'Ti_Fe_energy': Ti_Fe_energy,
        'Ti_Nd_distances': Ti_Nd_distances,
        'Ti_Fe_distances': Ti_Fe_distances,
        'closest_Nd_distance': min(Ti_Nd_distances) if Ti_Nd_distances else np.inf,
        'closest_Fe_distance': min(Ti_Fe_distances) if Ti_Fe_distances else np.inf
    }
