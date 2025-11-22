"""
Utilidades para crear y manipular la estructura cristalina 3D.

Este módulo maneja las coordenadas 3D reales de la estructura NdFe12.

Convención de valores en atom_types:
    0 = Fe (Hierro) - puede moverse (96 posiciones candidatas)
    1 = Nd (Neodimio) - FIJO (16 átomos)
    2 = Ti (Titanio) - sustituye a 8 Fe (lo que optimizamos)
"""

import numpy as np
from typing import Tuple, Optional


def get_Nd_positions_fijas() -> np.ndarray:
    """
    Retorna las posiciones fijas de los 16 átomos de Nd.

    Estas coordenadas son extraídas del notebook punto3.ipynb.

    Returns:
        np.ndarray: Array de shape (16, 3) con coordenadas (x, y, z) en Ångströms

    Examples:
        >>> Nd_pos = get_Nd_positions_fijas()
        >>> Nd_pos.shape
        (16, 3)
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

    positions = np.array([list(map(float, line.split()))
                         for line in Nd_coords_text.strip().split('\n')],
                        dtype=np.float32)

    assert positions.shape == (16, 3), f"Esperadas 16 posiciones Nd, encontradas {positions.shape}"

    return positions


def get_Fe_positions_all() -> np.ndarray:
    """
    Retorna TODAS las posiciones candidatas de Fe (96 posiciones).

    Estas son las posiciones donde potencialmente pueden estar átomos de Fe o Ti.

    Returns:
        np.ndarray: Array de shape (96, 3) con coordenadas (x, y, z) en Ångströms

    Note:
        El notebook original tiene 102 posiciones de Fe, pero usaremos solo 96
        para ser consistentes con el enunciado del problema (96 sitios candidatos).

    Examples:
        >>> Fe_pos = get_Fe_positions_all()
        >>> Fe_pos.shape
        (96, 3)
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
7.398000 5.484000 2.404000"""

    positions = np.array([list(map(float, line.split()))
                         for line in Fe_coords_text.strip().split('\n')],
                        dtype=np.float32)

    # Tomar solo las primeras 96 posiciones
    positions = positions[:96]

    assert positions.shape == (96, 3), f"Esperadas 96 posiciones Fe, encontradas {positions.shape}"

    return positions


def crear_configuracion_inicial(seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Crea la configuración inicial del sistema con 8 Ti aleatorios.

    Configuración:
        - 16 Nd fijos (coordenadas específicas)
        - 96 posiciones candidatas para Fe/Ti
        - 8 Ti en posiciones aleatorias (de las 96)
        - 88 Fe en el resto

    Args:
        seed: Semilla para reproducibilidad (opcional)

    Returns:
        Tupla (all_positions, atom_types, Ti_indices, Fe_indices, Nd_positions):
            - all_positions: np.ndarray shape (112, 3) - TODAS las coordenadas (16 Nd + 96 Fe/Ti)
            - atom_types: np.ndarray shape (112,) dtype=int8
                         Valores: 0=Fe, 1=Nd, 2=Ti
            - Ti_indices: np.ndarray shape (8,) dtype=int32
                         Índices de los 8 átomos de Ti en all_positions
            - Fe_indices: np.ndarray shape (88,) dtype=int32
                         Índices de los 88 átomos de Fe en all_positions
            - Nd_positions: np.ndarray shape (16, 3) dtype=float32
                           Coordenadas (x, y, z) de los 16 átomos de Nd (constante)

    Note:
        all_positions está estructurado como:
        - Índices 0-15: Nd (fijos)
        - Índices 16-111: Fe/Ti (candidatos, 96 posiciones)

    Examples:
        >>> all_pos, types, Ti_idx, Fe_idx, Nd_pos = crear_configuracion_inicial(seed=42)
        >>> all_pos.shape
        (112, 3)
        >>> np.sum(types == 2)  # Contar Ti
        8
    """
    if seed is not None:
        np.random.seed(seed)

    # Obtener posiciones fijas
    Nd_positions = get_Nd_positions_fijas()  # (16, 3)
    Fe_all_positions = get_Fe_positions_all()  # (96, 3)

    # Combinar todas las posiciones: [Nd (0-15), Fe/Ti (16-111)]
    all_positions = np.vstack([Nd_positions, Fe_all_positions]).astype(np.float32)

    # Inicializar tipos de átomo
    atom_types = np.zeros(112, dtype=np.int8)

    # Asignar Nd (índices 0-15)
    atom_types[0:16] = 1  # Nd = 1

    # Asignar Fe (índices 16-111, por ahora todo Fe)
    atom_types[16:112] = 0  # Fe = 0

    # Elegir 8 posiciones aleatorias de las 96 candidatas para Ti
    # Los índices candidatos van de 16 a 111 (96 posiciones)
    indices_candidatos = np.arange(16, 112)
    Ti_indices = np.random.choice(indices_candidatos, size=8, replace=False).astype(np.int32)

    # Marcar como Ti
    atom_types[Ti_indices] = 2  # Ti = 2

    # Obtener índices de Fe (todos los candidatos que NO son Ti)
    Fe_indices = np.array([i for i in indices_candidatos if i not in Ti_indices], dtype=np.int32)

    # Verificaciones
    assert len(all_positions) == 112, f"Esperadas 112 posiciones totales, encontradas {len(all_positions)}"
    assert np.sum(atom_types == 1) == 16, f"Esperados 16 Nd, encontrados {np.sum(atom_types == 1)}"
    assert np.sum(atom_types == 2) == 8, f"Esperados 8 Ti, encontrados {np.sum(atom_types == 2)}"
    assert np.sum(atom_types == 0) == 88, f"Esperados 88 Fe, encontrados {np.sum(atom_types == 0)}"
    assert len(Ti_indices) == 8, f"Esperados 8 índices Ti, encontrados {len(Ti_indices)}"
    assert len(Fe_indices) == 88, f"Esperados 88 índices Fe, encontrados {len(Fe_indices)}"

    return all_positions, atom_types, Ti_indices, Fe_indices, Nd_positions


def get_Ti_indices_from_types(atom_types: np.ndarray) -> np.ndarray:
    """
    Extrae los índices de los átomos de Ti desde atom_types.

    Args:
        atom_types: Array (112,) con tipos de átomos

    Returns:
        np.ndarray: Array con índices donde atom_types == 2 (Ti)

    Examples:
        >>> types = np.array([0, 0, 2, 1, 2, 0])
        >>> get_Ti_indices_from_types(types)
        array([2, 4])
    """
    return np.where(atom_types == 2)[0].astype(np.int32)
