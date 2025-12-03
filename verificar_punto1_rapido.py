"""Verificación rápida de punto1 después de correcciones"""

import sys
sys.path.insert(0, '/home/riemannintegrable/universidad/cadenas_de_markov/parcial2')

from src.punto1 import create_grid_4x4, get_Fe_positions_with_coords, compute_total_energy
import numpy as np

print("="*70)
print("VERIFICACIÓN RÁPIDA PUNTO 1")
print("="*70)

# Crear grid
grid = create_grid_4x4()

print(f"\n✓ Grid creado: {grid.n_R} Nd, {grid.n_Fe_sites} Fe")
print(f"\nPosiciones de Nd (primeras 2):")
print(grid.R_positions[:2])

# Verificar coordenadas
Fe_coords = get_Fe_positions_with_coords(grid)
print(f"\nPosiciones de Fe (primeras 4):")
for i in range(4):
    print(f"  {i}: {Fe_coords[i]}")

# Verificar que las coordenadas están en Angstroms (no truncadas)
if isinstance(Fe_coords[1], tuple) and isinstance(Fe_coords[1][0], float):
    print(f"\n✓ Coordenadas son float: {Fe_coords[1]}")
else:
    print(f"\n✗ ERROR: Coordenadas no son float")

# Calcular energía de una configuración
grid.set_Ti_position(0)
energia = compute_total_energy(grid)
print(f"\n✓ Energía calculada para Ti en posición 0: {energia:.6f}")

# Verificar que la energía es razonable
if -100 < energia < 100:
    print(f"✓ Energía en rango razonable")
else:
    print(f"⚠ Energía fuera de rango esperado")

print("\n" + "="*70)
print("VERIFICACIÓN COMPLETADA")
print("="*70)
