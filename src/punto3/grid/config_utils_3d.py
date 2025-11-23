"""
Utilidades para crear y manipular configuraciones atómicas 3D.

Este módulo maneja la estructura cristalina NdFe12 en 3D (Punto 3).
A diferencia del Punto 2, aquí NO hay grilla discreta, sino posiciones
3D continuas (coordenadas reales en Angstroms).

Estructura de datos:
    - Fe_candidate_positions: (96, 3) - Posiciones candidatas FIJAS para Fe/Ti
    - Nd_positions: (16, 3) - Posiciones FIJAS de Nd
    - Ti_indices: (8,) - Índices de cuáles de los 96 Fe tienen Ti
    - Fe_indices: (88,) - Índices de los Fe que NO tienen Ti
    - atom_types: (112,) - Tipo de cada átomo (0=Fe, 1=Nd, 2=Ti)
    - all_positions: (112, 3) - Todas las posiciones concatenadas

Convención de valores en atom_types:
    0 = Fe (Hierro)
    1 = Nd (Neodimio) - FIJO
    2 = Ti (Titanio) - sustituye a Fe
"""

import numpy as np
from typing import Tuple, Optional


def crear_configuracion_inicial_3d(
    Fe_candidate_positions: np.ndarray,
    Nd_positions: np.ndarray,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Crea la configuración inicial del sistema 3D con 8 Ti aleatorios.

    Configuración:
        - 96 posiciones candidatas para Fe/Ti (FIJAS en el espacio)
        - 16 posiciones de Nd (FIJAS)
        - 8 Ti colocados aleatoriamente en 8 de las 96 posiciones
        - 88 Fe en las 88 posiciones restantes

    Args:
        Fe_candidate_positions: Array (96, 3) con coordenadas (x, y, z) de candidatos Fe
        Nd_positions: Array (16, 3) con coordenadas (x, y, z) de átomos Nd
        seed: Semilla para reproducibilidad (opcional)

    Returns:
        Tupla (atom_types, all_positions, Ti_indices, Fe_indices):
            - atom_types: np.ndarray shape (112,), dtype=int8
                         Valores: 0=Fe, 1=Nd, 2=Ti
                         Orden: [96 candidatos Fe/Ti] + [16 Nd]
            - all_positions: np.ndarray shape (112, 3), dtype=float64
                            Concatenación: [Fe_candidate_positions, Nd_positions]
            - Ti_indices: np.ndarray shape (8,), dtype=int
                         Índices (0-95) de cuáles candidatos tienen Ti
            - Fe_indices: np.ndarray shape (88,), dtype=int
                         Índices (0-95) de cuáles candidatos tienen Fe

    Examples:
        >>> Fe_pos = np.random.rand(96, 3)
        >>> Nd_pos = np.random.rand(16, 3)
        >>> atom_types, all_pos, Ti_idx, Fe_idx = crear_configuracion_inicial_3d(Fe_pos, Nd_pos, seed=42)
        >>> atom_types.shape
        (112,)
        >>> all_pos.shape
        (112, 3)
        >>> Ti_idx.shape
        (8,)
        >>> Fe_idx.shape
        (88,)
        >>> np.sum(atom_types == 2)  # Contar Ti
        8
        >>> np.sum(atom_types == 1)  # Contar Nd
        16
        >>> np.sum(atom_types == 0)  # Contar Fe
        88
    """
    # Validaciones
    assert Fe_candidate_positions.shape == (96, 3), \
        f"Fe_candidate_positions debe ser (96, 3), recibido {Fe_candidate_positions.shape}"
    assert Nd_positions.shape == (16, 3), \
        f"Nd_positions debe ser (16, 3), recibido {Nd_positions.shape}"

    if seed is not None:
        np.random.seed(seed)

    # Seleccionar 8 índices aleatorios de los 96 candidatos para Ti
    Ti_indices = np.random.choice(96, size=8, replace=False)
    Ti_indices = np.sort(Ti_indices)  # Ordenar para consistencia

    # Los demás son Fe
    all_candidate_indices = np.arange(96)
    Fe_indices = np.setdiff1d(all_candidate_indices, Ti_indices)

    # Crear array de tipos atómicos
    # Orden: [96 candidatos Fe/Ti] + [16 Nd]
    atom_types = np.zeros(112, dtype=np.int8)

    # Marcar Ti en sus posiciones (0-95)
    atom_types[Ti_indices] = 2  # Ti = 2

    # Marcar Nd en posiciones 96-111
    atom_types[96:112] = 1  # Nd = 1

    # El resto (Fe) ya son 0

    # Concatenar posiciones
    all_positions = np.vstack([Fe_candidate_positions, Nd_positions])

    # Verificaciones
    assert np.sum(atom_types == 0) == 88, f"Esperados 88 Fe, encontrados {np.sum(atom_types == 0)}"
    assert np.sum(atom_types == 1) == 16, f"Esperados 16 Nd, encontrados {np.sum(atom_types == 1)}"
    assert np.sum(atom_types == 2) == 8, f"Esperados 8 Ti, encontrados {np.sum(atom_types == 2)}"
    assert all_positions.shape == (112, 3), f"all_positions debe ser (112, 3)"

    return atom_types, all_positions, Ti_indices, Fe_indices


def aplicar_swap_3d(
    atom_types: np.ndarray,
    Ti_indices: np.ndarray,
    Fe_indices: np.ndarray,
    ti_idx: int,
    fe_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aplica un swap entre un Ti y un Fe en la configuración 3D.

    El swap intercambia los tipos atómicos, pero NO las posiciones
    (las posiciones son fijas en el espacio).

    Esta función NO modifica los arrays originales, retorna copias.

    Args:
        atom_types: Array (112,) con tipos atómicos (0=Fe, 1=Nd, 2=Ti)
        Ti_indices: Array (8,) con índices (0-95) de átomos de Ti
        Fe_indices: Array (88,) con índices (0-95) de átomos de Fe
        ti_idx: Índice en Ti_indices del Ti a intercambiar (0-7)
        fe_idx: Índice en Fe_indices del Fe a intercambiar (0-87)

    Returns:
        Tupla (atom_types_new, Ti_indices_new, Fe_indices_new) con el swap aplicado

    Note:
        - El swap cambia qué posición candidata (0-95) tiene Ti vs Fe
        - Las posiciones espaciales (x,y,z) NO cambian
        - Esta función crea copias para evitar modificar el estado original

    Examples:
        >>> atom_types = np.array([2, 0, 0, 1], dtype=np.int8)  # Ti, Fe, Fe, Nd
        >>> Ti_idx = np.array([0])
        >>> Fe_idx = np.array([1, 2])
        >>> atom_types_new, Ti_new, Fe_new = aplicar_swap_3d(atom_types, Ti_idx, Fe_idx, 0, 1)
        >>> atom_types_new
        array([0, 0, 2, 1], dtype=int8)  # Fe, Fe, Ti, Nd
        >>> Ti_new
        array([2])  # Ahora Ti está en posición 2
        >>> Fe_new
        array([0, 1])  # Ahora Fe están en posiciones 0, 1
    """
    # Copiar arrays
    atom_types_new = atom_types.copy()
    Ti_new = Ti_indices.copy()
    Fe_new = Fe_indices.copy()

    # Obtener posiciones globales (0-95) que se van a intercambiar
    ti_global_idx = Ti_indices[ti_idx]  # Posición candidata que tiene Ti
    fe_global_idx = Fe_indices[fe_idx]  # Posición candidata que tiene Fe

    # Swap en atom_types
    atom_types_new[ti_global_idx] = 0  # Ahora es Fe
    atom_types_new[fe_global_idx] = 2  # Ahora es Ti

    # Actualizar índices
    Ti_new[ti_idx] = fe_global_idx  # Ti ahora está donde estaba Fe
    Fe_new[fe_idx] = ti_global_idx  # Fe ahora está donde estaba Ti

    # Mantener ordenados para consistencia
    Ti_new = np.sort(Ti_new)
    Fe_new = np.sort(Fe_new)

    return atom_types_new, Ti_new, Fe_new


def get_Ti_positions_3d(
    all_positions: np.ndarray,
    Ti_indices: np.ndarray
) -> np.ndarray:
    """
    Extrae las posiciones 3D de los átomos de Ti.

    Args:
        all_positions: Array (112, 3) con todas las posiciones
        Ti_indices: Array (8,) con índices (0-95) de átomos de Ti

    Returns:
        np.ndarray: Array (8, 3) con coordenadas (x, y, z) de Ti

    Examples:
        >>> all_pos = np.random.rand(112, 3)
        >>> Ti_idx = np.array([0, 5, 10, 20, 30, 40, 50, 60])
        >>> Ti_pos = get_Ti_positions_3d(all_pos, Ti_idx)
        >>> Ti_pos.shape
        (8, 3)
    """
    return all_positions[Ti_indices]


def get_Fe_positions_3d(
    all_positions: np.ndarray,
    Fe_indices: np.ndarray
) -> np.ndarray:
    """
    Extrae las posiciones 3D de los átomos de Fe.

    Args:
        all_positions: Array (112, 3) con todas las posiciones
        Fe_indices: Array (88,) con índices (0-95) de átomos de Fe

    Returns:
        np.ndarray: Array (88, 3) con coordenadas (x, y, z) de Fe

    Examples:
        >>> all_pos = np.random.rand(112, 3)
        >>> Fe_idx = np.arange(88)
        >>> Fe_pos = get_Fe_positions_3d(all_pos, Fe_idx)
        >>> Fe_pos.shape
        (88, 3)
    """
    return all_positions[Fe_indices]


def get_Nd_positions_3d(all_positions: np.ndarray) -> np.ndarray:
    """
    Extrae las posiciones 3D de los átomos de Nd (siempre posiciones 96-111).

    Args:
        all_positions: Array (112, 3) con todas las posiciones

    Returns:
        np.ndarray: Array (16, 3) con coordenadas (x, y, z) de Nd

    Examples:
        >>> all_pos = np.random.rand(112, 3)
        >>> Nd_pos = get_Nd_positions_3d(all_pos)
        >>> Nd_pos.shape
        (16, 3)
    """
    return all_positions[96:112]


def validar_configuracion(
    atom_types: np.ndarray,
    Ti_indices: np.ndarray,
    Fe_indices: np.ndarray
) -> bool:
    """
    Valida que la configuración sea consistente.

    Args:
        atom_types: Array (112,) con tipos atómicos
        Ti_indices: Array (8,) con índices de Ti
        Fe_indices: Array (88,) con índices de Fe

    Returns:
        bool: True si la configuración es válida

    Raises:
        AssertionError: Si hay inconsistencias
    """
    # Validar tamaños
    assert atom_types.shape == (112,), f"atom_types debe ser (112,), es {atom_types.shape}"
    assert Ti_indices.shape == (8,), f"Ti_indices debe ser (8,), es {Ti_indices.shape}"
    assert Fe_indices.shape == (88,), f"Fe_indices debe ser (88,), es {Fe_indices.shape}"

    # Validar conteos
    assert np.sum(atom_types == 0) == 88, f"Debe haber 88 Fe"
    assert np.sum(atom_types == 1) == 16, f"Debe haber 16 Nd"
    assert np.sum(atom_types == 2) == 8, f"Debe haber 8 Ti"

    # Validar que Ti_indices apuntan a Ti en atom_types
    for ti_idx in Ti_indices:
        assert atom_types[ti_idx] == 2, f"atom_types[{ti_idx}] debe ser Ti (2)"

    # Validar que Fe_indices apuntan a Fe en atom_types
    for fe_idx in Fe_indices:
        assert atom_types[fe_idx] == 0, f"atom_types[{fe_idx}] debe ser Fe (0)"

    # Validar que Ti_indices y Fe_indices no se solapan y cubren 0-95
    all_candidate_indices = np.sort(np.concatenate([Ti_indices, Fe_indices]))
    assert np.array_equal(all_candidate_indices, np.arange(96)), \
        "Ti_indices y Fe_indices deben cubrir exactamente 0-95 sin solapamiento"

    # Validar que posiciones 96-111 son Nd
    assert np.all(atom_types[96:112] == 1), "Posiciones 96-111 deben ser Nd"

    return True
