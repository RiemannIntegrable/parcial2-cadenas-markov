"""
Utilidades para la estructura cristalina 3D de NdFe₁₂

Este módulo provee las coordenadas atómicas reales de la estructura cristalina
y funciones para crear configuraciones iniciales con Ti aleatorios.
"""

import numpy as np
from typing import Tuple, Optional


def get_Nd_positions_fixed() -> np.ndarray:
    """
    Retorna las 16 posiciones FIJAS de átomos de Neodimio (Nd).

    Estas coordenadas fueron extraídas de la estructura cristalina real de NdFe₁₂
    y permanecen fijas durante la optimización.

    Returns:
        Array (16, 3) con coordenadas (x, y, z) en Angstroms
    """
    Nd_coords_text = """8.592000 8.592000 0.000000
0.000000 8.592000 2.404000
8.592000 0.000000 2.404000
0.000000 0.000000 0.000000
12.888000 4.296000 0.000000
4.296000 4.296000 2.404000
12.888000 12.888000 2.404000
4.296000 12.888000 0.000000
4.296000 12.888000 2.404000
12.888000 12.888000 0.000000
4.296000 4.296000 0.000000
12.888000 4.296000 2.404000
0.000000 0.000000 2.404000
8.592000 0.000000 0.000000
0.000000 8.592000 0.000000
8.592000 8.592000 2.404000"""

    Nd_positions = np.array([
        list(map(float, line.split()))
        for line in Nd_coords_text.strip().split('\n')
    ], dtype=np.float32)

    assert Nd_positions.shape == (16, 3), f"Expected (16, 3), got {Nd_positions.shape}"

    return Nd_positions


def get_Fe_positions_all() -> np.ndarray:
    """
    Retorna las 96 posiciones candidatas de átomos de Hierro (Fe).

    Estas son las posiciones donde pueden ubicarse átomos de Fe o Ti.
    El dataset original tiene 102 posiciones, pero según el enunciado del problema
    debemos usar solo 96 sitios candidatos.

    Returns:
        Array (96, 3) con coordenadas (x, y, z) en Angstroms

    Note:
        Se truncan las primeras 96 posiciones del dataset completo de 102.
    """
    Fe_coords_text = """2.502960 8.592000 1.202000
6.089040 0.000000 1.202000
8.592000 2.502960 1.202000
0.000000 6.089040 1.202000
11.094960 4.296000 1.202000
1.793040 12.888000 1.202000
12.888000 11.094960 1.202000
4.296000 1.793040 1.202000
6.089040 8.592000 1.202000
2.502960 0.000000 1.202000
8.592000 6.089040 1.202000
0.000000 2.502960 1.202000
1.793040 4.296000 1.202000
11.094960 12.888000 1.202000
12.888000 1.793040 1.202000
4.296000 11.094960 1.202000
4.296000 0.000000 0.000000
12.888000 8.592000 0.000000
8.592000 4.296000 0.000000
0.000000 12.888000 0.000000
8.592000 12.888000 0.000000
0.000000 4.296000 0.000000
12.888000 0.000000 0.000000
4.296000 8.592000 0.000000
4.296000 8.592000 2.404000
12.888000 0.000000 2.404000
8.592000 12.888000 2.404000
0.000000 4.296000 2.404000
8.592000 4.296000 2.404000
0.000000 12.888000 2.404000
12.888000 8.592000 2.404000
4.296000 0.000000 2.404000
5.484000 5.484000 0.000000
11.700000 11.700000 0.000000
1.206000 11.700000 0.000000
7.398000 1.206000 0.000000
7.398000 1.206000 2.404000
1.206000 11.700000 2.404000
11.700000 11.700000 2.404000
5.484000 5.484000 2.404000
11.700000 5.484000 2.404000
5.484000 11.700000 2.404000
7.398000 7.398000 2.404000
1.206000 1.206000 2.404000
1.206000 1.206000 0.000000
7.398000 7.398000 0.000000
5.484000 11.700000 0.000000
11.700000 5.484000 0.000000
15.390960 6.089040 1.202000
10.385040 1.793040 1.202000
6.807000 11.094960 1.202000
1.793040 6.807000 1.202000
10.385040 11.094960 1.202000
1.793040 15.390960 1.202000
6.807000 1.793040 1.202000
15.390960 6.807000 1.202000
1.793040 10.385040 1.202000
6.807000 6.089040 1.202000
15.390960 1.793040 1.202000
11.094960 6.807000 1.202000
6.089040 1.793040 1.202000
11.094960 10.385040 1.202000
2.502960 6.807000 1.202000
6.807000 2.502960 1.202000
11.094960 15.390960 1.202000
6.089040 6.807000 1.202000
2.502960 10.385040 1.202000
10.385040 2.502960 1.202000
15.390960 11.094960 1.202000
10.385040 6.089040 1.202000
6.807000 15.390960 1.202000
2.502960 1.793040 1.202000
15.390960 2.502960 1.202000
6.089040 10.385040 1.202000
12.888000 8.592000 2.404000
4.296000 0.000000 0.000000
12.888000 0.000000 2.404000
4.296000 8.592000 0.000000
4.296000 8.592000 2.404000
12.888000 0.000000 0.000000
8.592000 12.888000 0.000000
0.000000 4.296000 0.000000
8.592000 4.296000 2.404000
0.000000 12.888000 2.404000
12.888000 8.592000 0.000000
4.296000 0.000000 2.404000
1.206000 7.398000 2.404000
7.398000 11.700000 2.404000
5.484000 1.206000 2.404000
11.700000 7.398000 2.404000
11.700000 7.398000 0.000000
5.484000 1.206000 0.000000
7.398000 11.700000 0.000000
1.206000 7.398000 0.000000
1.206000 5.484000 2.404000
7.398000 5.484000 2.404000
5.484000 7.398000 2.404000
11.700000 1.206000 2.404000"""

    Fe_positions = np.array([
        list(map(float, line.split()))
        for line in Fe_coords_text.strip().split('\n')
    ], dtype=np.float32)

    # Truncar a las primeras 96 posiciones según enunciado del problema
    Fe_positions = Fe_positions[:96]

    assert Fe_positions.shape == (96, 3), f"Expected (96, 3), got {Fe_positions.shape}"

    return Fe_positions


def get_Ti_indices_from_types(atom_types: np.ndarray) -> np.ndarray:
    """
    Extrae los índices globales de los átomos de Ti desde el array atom_types.

    Esta función es CRÍTICA para evitar el bug de pérdida de átomos de Ti.
    Siempre usa esta función para extraer los índices de Ti después de la optimización,
    NO confíes en el array Ti_indices retornado por el algoritmo.

    Args:
        atom_types: Array (N,) con tipos de átomos [0=Fe, 1=Nd, 2=Ti]

    Returns:
        Array con índices globales donde atom_types == 2 (Ti)

    Example:
        >>> atom_types = np.array([0, 1, 2, 0, 2, 1])  # 2 Ti en posiciones 2 y 4
        >>> get_Ti_indices_from_types(atom_types)
        array([2, 4], dtype=int32)
    """
    Ti_indices = np.where(atom_types == 2)[0].astype(np.int32)
    return Ti_indices


def crear_configuracion_inicial_3d(
    n_Ti: int = 8,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Crea una configuración inicial aleatoria del sistema cristalino 3D.

    El sistema consta de:
    - 16 átomos de Nd (fijos)
    - 88 átomos de Fe
    - 8 átomos de Ti (colocados aleatoriamente en posiciones de Fe)

    Args:
        n_Ti: Número de átomos de Ti (default: 8)
        seed: Semilla para reproducibilidad (opcional)

    Returns:
        Tupla (all_positions, atom_types, Ti_indices, Nd_positions) donde:
        - all_positions: Array (112, 3) con todas las coordenadas atómicas
        - atom_types: Array (112,) con tipos [0=Fe, 1=Nd, 2=Ti]
        - Ti_indices: Array (8,) con índices globales de Ti
        - Nd_positions: Array (16, 3) con posiciones fijas de Nd

    Example:
        >>> all_pos, types, Ti_idx, Nd_pos = crear_configuracion_inicial_3d(seed=42)
        >>> all_pos.shape
        (112, 3)
        >>> np.sum(types == 2)  # Contar Ti
        8
    """
    if seed is not None:
        np.random.seed(seed)

    # Cargar posiciones cristalográficas
    Nd_positions = get_Nd_positions_fixed()  # (16, 3)
    Fe_positions = get_Fe_positions_all()    # (96, 3)

    # Seleccionar aleatoriamente n_Ti posiciones de las 96 para Ti
    all_Fe_indices = np.arange(96)
    Ti_local_indices = np.random.choice(all_Fe_indices, size=n_Ti, replace=False)
    Ti_local_indices = np.sort(Ti_local_indices)  # Ordenar para facilitar debugging

    # Crear array atom_types
    # Estructura: [Fe (88 átomos) | Nd (16 átomos) | Ti (8 átomos)]
    # Índices:    [0..87         | 88..103        | 104..111     ]

    Fe_mask = np.ones(96, dtype=bool)
    Fe_mask[Ti_local_indices] = False

    Fe_only_positions = Fe_positions[Fe_mask]  # (88, 3)
    Ti_positions = Fe_positions[Ti_local_indices]  # (8, 3)

    # Construir all_positions: concatenar en orden Fe | Nd | Ti
    all_positions = np.vstack([
        Fe_only_positions,  # 0..87
        Nd_positions,        # 88..103
        Ti_positions         # 104..111
    ]).astype(np.float32)

    # Construir atom_types
    atom_types = np.concatenate([
        np.zeros(88, dtype=np.int32),   # Fe
        np.ones(16, dtype=np.int32),    # Nd
        np.full(8, 2, dtype=np.int32)   # Ti
    ])

    # Índices globales de Ti en all_positions
    Ti_indices = np.arange(104, 112, dtype=np.int32)  # [104, 105, ..., 111]

    # Verificaciones de sanidad
    assert all_positions.shape == (112, 3), f"all_positions shape: {all_positions.shape}"
    assert atom_types.shape == (112,), f"atom_types shape: {atom_types.shape}"
    assert len(Ti_indices) == n_Ti, f"Ti_indices length: {len(Ti_indices)}"
    assert np.sum(atom_types == 0) == 88, f"Fe count: {np.sum(atom_types == 0)}"
    assert np.sum(atom_types == 1) == 16, f"Nd count: {np.sum(atom_types == 1)}"
    assert np.sum(atom_types == 2) == n_Ti, f"Ti count: {np.sum(atom_types == 2)}"

    return all_positions, atom_types, Ti_indices, Nd_positions


def print_system_info(all_positions: np.ndarray, atom_types: np.ndarray):
    """
    Imprime información útil sobre el sistema cristalino.

    Args:
        all_positions: Array (N, 3) con posiciones
        atom_types: Array (N,) con tipos de átomos
    """
    n_Fe = np.sum(atom_types == 0)
    n_Nd = np.sum(atom_types == 1)
    n_Ti = np.sum(atom_types == 2)

    print("="*70)
    print("INFORMACIÓN DEL SISTEMA CRISTALINO 3D")
    print("="*70)
    print(f"  Total de átomos:     {len(all_positions)}")
    print(f"  Átomos de Fe (0):    {n_Fe}")
    print(f"  Átomos de Nd (1):    {n_Nd} (fijos)")
    print(f"  Átomos de Ti (2):    {n_Ti} (a optimizar)")
    print(f"\n  Rango X: [{all_positions[:, 0].min():.3f}, {all_positions[:, 0].max():.3f}] Å")
    print(f"  Rango Y: [{all_positions[:, 1].min():.3f}, {all_positions[:, 1].max():.3f}] Å")
    print(f"  Rango Z: [{all_positions[:, 2].min():.3f}, {all_positions[:, 2].max():.3f}] Å")
    print("="*70)
