# Punto 3: Optimización en Estructura Cristalina 3D (NdFe₁₂)

## Descripción

Este módulo adapta el algoritmo de Simulated Annealing del Punto 2 para trabajar con la estructura cristalina real de NdFe₁₂ en 3D.

### Problema

Encontrar la configuración óptima de **8 átomos de Ti** que sustituyen a 8 átomos de Fe en una estructura con:
- **16 átomos de Nd** (Neodimio) - posiciones fijas
- **96 posiciones candidatas de Fe** (Hierro)
- **Espacio de búsqueda**: C(96, 8) ≈ 1.86 × 10¹¹ configuraciones

### Diferencias con Punto 2

| Aspecto | Punto 2 (2D) | Punto 3 (3D) |
|---------|--------------|--------------|
| **Estructura** | Grilla 10×10 discreta | Coordenadas 3D reales (Å) |
| **Átomos totales** | 100 | 112 |
| **Posiciones Nd** | 16 (núcleo 4×4 central) | 16 (coordenadas específicas) |
| **Posiciones Fe candidatas** | 76 | 96 |
| **Distancias** | Euclidianas 2D (enteras) | Euclidianas 3D (float) |
| **Representación** | Array 2D (grid) | Arrays de coordenadas 3D |

## Estructura de Módulos

```
punto3/
├── morse/                      # Potencial de Morse (idéntico a punto2)
│   └── potential_numba.py      # Con distancia_3d en lugar de distancia_2d
│
├── crystal/                    # Reemplazo de 'grid' para 3D
│   ├── crystal_utils.py        # Manejo de coordenadas 3D
│   └── energy_numba.py         # Cálculo de energía optimizado para 3D
│
├── optimization/               # Simulated Annealing adaptado
│   ├── sa_numba.py            # Core del algoritmo con Numba
│   ├── cooling_schedules.py   # Esquemas de enfriamiento
│   └── parallel_runs.py       # Ejecución paralela con joblib
│
├── analysis/                   # Análisis espacial 3D
│   └── spatial_analysis_3d.py  # Métricas de distribución en 3D
│
└── visualization/              # Gráficas 3D con Plotly
    └── plotting_3d.py          # Visualización de la estructura
```

## Optimizaciones Clave Mantenidas

### 1. Cálculo Incremental de ΔE (O(N) vs O(N²))

```python
# En lugar de recalcular toda la energía después de cada swap:
# E_total = sum_over_all_pairs(U_morse)  # O(N²)

# Solo calculamos la diferencia incremental:
ΔE = (U_new_Ti + U_new_Fe) - (U_old_Ti + U_old_Fe)  # O(N)
```

**Speedup**: 25× sin Numba, ~1000× con Numba

### 2. Compilación JIT con Numba

Todas las funciones críticas están decoradas con `@njit(fastmath=True, cache=True)`:
- `compute_total_energy_fast_3d`
- `compute_delta_E_swap_fast_3d`
- `simulated_annealing_core_logarithmic_3d`

### 3. Ejecución Paralela con Joblib

Múltiples runs independientes ejecutados en paralelo aprovechando todos los cores del CPU.

## Uso

### Instalación de Dependencias

```bash
pip install numpy numba joblib plotly
```

### Ejemplo Básico

```python
from src.punto3.morse import preparar_morse_params_array
from src.punto3.crystal import crear_configuracion_inicial, compute_total_energy_fast_3d
from src.punto3.optimization import simulated_annealing_logarithmic_3d

# 1. Preparar parámetros
morse_params = preparar_morse_params_array()

# 2. Crear configuración inicial
all_positions, atom_types, Ti_indices, Fe_indices, _ = crear_configuracion_inicial(seed=42)

# 3. Ejecutar Simulated Annealing
atom_types_best, Ti_indices_best, history = simulated_annealing_logarithmic_3d(
    all_positions,
    atom_types,
    Ti_indices,
    Fe_indices,
    morse_params,
    c=3000,
    t0=2,
    max_iterations=5_000_000,
    seed=42
)

# 4. Analizar resultados
print(f"Energía óptima: {history['energy_best']:.6f}")
print(f"Convergió en iteración: {history['iterations_to_best']:,}")
```

### Ejecución de Múltiples Runs

```python
from src.punto3.optimization.parallel_runs import (
    ejecutar_multiples_runs_logarithmic_3d,
    get_best_run
)

# Ejecutar 64 runs en paralelo
resultados = ejecutar_multiples_runs_logarithmic_3d(
    n_runs=64,
    c=3000,
    t0=2,
    max_iterations=5_000_000,
    morse_params=morse_params,
    n_jobs=-1  # Usar todos los cores
)

# Obtener mejor resultado
mejor = get_best_run(resultados)
print(f"Mejor energía de 64 runs: {mejor['energia_final']:.6f}")
```

### Visualización

```python
from src.punto3.visualization import plot_crystal_configuration_3d

# Visualizar configuración óptima
fig = plot_crystal_configuration_3d(
    all_positions,
    atom_types_best,
    Nd_positions,
    Ti_indices_best,
    energia=history['energy_best'],
    title="Configuración Óptima"
)
fig.show()
```

## Notebook

Ver `notebooks/punto3.ipynb` para un análisis completo con:
- Visualización 3D de la estructura
- Ejecución de múltiples runs
- Análisis de convergencia
- Métricas espaciales de distribución de Ti
- Comparación de resultados

## Parámetros Recomendados

Para garantizar convergencia (Teorema de Hajek):

```python
c = 3000        # Constante ≥ profundidad de barreras
t0 = 2          # Offset temporal
max_iter = 5e6  # 5 millones de iteraciones
N_RUNS = 64     # Múltiples runs independientes
```

**Tiempo estimado**: ~2-3 minutos por run en CPU moderno (total ~3-5 min para 64 runs en paralelo)

## Referencias

- **Hajek, B. (1988)**. "Cooling Schedules for Optimal Annealing". _Mathematics of Operations Research_.
- **Kirkpatrick, S., et al. (1983)**. "Optimization by Simulated Annealing". _Science_.
