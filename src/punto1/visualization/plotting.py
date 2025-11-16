"""
Funciones de visualización para análisis de resultados.

Este módulo proporciona utilidades para visualizar:
    - Grillas con configuraciones de átomos
    - Evolución de energía en SA
    - Distribuciones de energía
    - Comparaciones entre métodos
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Optional, Dict, List
from ..grid import Grid2D


# Configuración de colores para átomos
ATOM_COLORS = {
    'Nd': '#e74c3c',  # Rojo para Neodimio
    'Fe': '#3498db',  # Azul para Hierro
    'Ti': '#2ecc71'   # Verde para Titanio
}

ATOM_SIZES = {
    'Nd': 200,
    'Fe': 150,
    'Ti': 180
}


def plot_grid_configuration(
    grid: Grid2D,
    title: str = "Configuración de la Grilla",
    highlight_Ti: bool = True,
    show_indices: bool = False,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Visualiza una configuración de grilla 2D con átomos coloreados.

    Args:
        grid: Grilla 2D a visualizar
        title: Título de la figura
        highlight_Ti: Si resaltar el átomo de Ti con borde
        show_indices: Si mostrar índices de posiciones Fe
        ax: Axes existente (None = crear nuevo)

    Returns:
        Axes de matplotlib

    Examples:
        >>> from src.grid import create_grid_4x4
        >>> grid = create_grid_4x4()
        >>> grid.set_Ti_position(0)
        >>> ax = plot_grid_configuration(grid)
        >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    atoms = grid.get_all_atoms()

    # Plotear cada átomo
    for pos, atom_type in atoms:
        x, y = pos
        color = ATOM_COLORS[atom_type]
        size = ATOM_SIZES[atom_type]

        # Scatter plot
        ax.scatter(x, y, c=color, s=size, alpha=0.8,
                   edgecolors='black', linewidths=2 if atom_type == 'Ti' and highlight_Ti else 1,
                   label=atom_type, zorder=3)

        # Etiqueta del tipo
        ax.text(x, y, atom_type, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=4)

    # Mostrar índices de posiciones Fe si se solicita
    if show_indices:
        for i, pos in enumerate(grid.Fe_positions):
            x, y = pos
            # Offset para que no se superponga con la etiqueta del átomo
            ax.text(x + 0.3, y + 0.3, str(i), fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Configuración de la grilla
    ax.set_xlim(-0.5, grid.size[1] - 0.5)
    ax.set_ylim(-0.5, grid.size[0] - 0.5)
    ax.set_aspect('equal')

    # Grilla de fondo
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(range(grid.size[1]))
    ax.set_yticks(range(grid.size[0]))

    # Etiquetas
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Leyenda única (eliminar duplicados)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper right')

    return ax


def plot_energy_evolution(
    energy_history: List[float],
    temperature_history: Optional[List[float]] = None,
    optimal_energy: Optional[float] = None,
    title: str = "Evolución de Simulated Annealing",
    figsize: tuple = (12, 8)
) -> plt.Figure:
    """
    Visualiza la evolución de energía (y temperatura) durante SA.

    Args:
        energy_history: Lista de energías en cada iteración
        temperature_history: Lista de temperaturas (opcional)
        optimal_energy: Energía óptima conocida (línea de referencia)
        title: Título de la figura
        figsize: Tamaño de la figura

    Returns:
        Figure de matplotlib

    Examples:
        >>> result = simulated_annealing(grid, max_iter=1000)
        >>> fig = plot_energy_evolution(result['energy_history'],
        ...                             result['temperature_history'])
        >>> plt.show()
    """
    if temperature_history is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    iterations = range(len(energy_history))

    # Plot de energía
    ax1.plot(iterations, energy_history, alpha=0.7, linewidth=0.8,
             color='#3498db', label='Energía actual')

    if optimal_energy is not None:
        ax1.axhline(optimal_energy, color='#e74c3c', linestyle='--',
                    linewidth=2, label='Óptimo global (fuerza bruta)')

    ax1.set_ylabel('Energía E(s)', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot de temperatura (si está disponible)
    if temperature_history is not None:
        ax2.semilogy(iterations, temperature_history, color='#e67e22',
                     linewidth=1.5, label='Temperatura')
        ax2.set_xlabel('Iteración', fontsize=12)
        ax2.set_ylabel('Temperatura T (log scale)', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # Anotar fases
        n_iter = len(iterations)
        ax2.axvspan(0, n_iter * 0.3, alpha=0.1, color='red', label='Exploración')
        ax2.axvspan(n_iter * 0.7, n_iter, alpha=0.1, color='blue', label='Explotación')

    else:
        ax1.set_xlabel('Iteración', fontsize=12)

    plt.tight_layout()
    return fig


def plot_energy_distribution(
    all_energies: np.ndarray,
    best_energy: float,
    title: str = "Distribución de Energías (Todas las Configuraciones)",
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Histograma de energías de todas las configuraciones posibles.

    Útil para visualizar el paisaje de energía del problema.

    Args:
        all_energies: Array con energías de todas las configuraciones
        best_energy: Energía óptima (marcador vertical)
        title: Título de la figura
        figsize: Tamaño de la figura

    Returns:
        Figure de matplotlib

    Examples:
        >>> result_bf = brute_force_search(grid)
        >>> fig = plot_energy_distribution(result_bf['all_energies'],
        ...                                result_bf['best_energy'])
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Histograma
    n, bins, patches = ax.hist(all_energies, bins=20, alpha=0.7,
                                color='#3498db', edgecolor='black')

    # Marcar el óptimo
    ax.axvline(best_energy, color='#e74c3c', linestyle='--',
               linewidth=3, label=f'Óptimo global: {best_energy:.4f}')

    # Marcar la media
    mean_energy = np.mean(all_energies)
    ax.axvline(mean_energy, color='#f39c12', linestyle=':',
               linewidth=2, label=f'Media: {mean_energy:.4f}')

    ax.set_xlabel('Energía', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_comparison_bar(
    methods: List[str],
    energies: List[float],
    title: str = "Comparación de Energías entre Métodos",
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Gráfica de barras comparando energías de diferentes métodos.

    Args:
        methods: Lista de nombres de métodos
        energies: Lista de energías correspondientes
        title: Título de la figura
        figsize: Tamaño de la figura

    Returns:
        Figure de matplotlib

    Examples:
        >>> methods = ['Fuerza Bruta', 'SA (α=0.95)', 'SA (α=0.90)']
        >>> energies = [-10.5, -10.5, -10.3]
        >>> fig = plot_comparison_bar(methods, energies)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    x_pos = np.arange(len(methods))

    bars = ax.bar(x_pos, energies, color=colors[:len(methods)],
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    # Anotar valores en las barras
    for i, (bar, energy) in enumerate(zip(bars, energies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{energy:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylabel('Energía', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_acceptance_rate(
    energy_history: List[float],
    window_size: int = 100,
    title: str = "Evolución de la Tasa de Aceptación",
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Calcula y grafica la tasa de aceptación en ventanas móviles.

    Args:
        energy_history: Historia de energías
        window_size: Tamaño de la ventana móvil
        title: Título de la figura
        figsize: Tamaño de la figura

    Returns:
        Figure de matplotlib

    Examples:
        >>> result = simulated_annealing(grid, max_iter=5000)
        >>> fig = plot_acceptance_rate(result['energy_history'])
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Detectar aceptaciones (cambios de energía)
    acceptances = [
        1 if energy_history[i] != energy_history[i - 1] else 0
        for i in range(1, len(energy_history))
    ]

    # Calcular tasa en ventanas móviles
    if len(acceptances) >= window_size:
        accept_rates = []
        for i in range(len(acceptances) - window_size + 1):
            rate = np.mean(acceptances[i:i + window_size])
            accept_rates.append(rate)

        iterations = range(window_size, len(energy_history))
        ax.plot(iterations, accept_rates, color='#3498db', linewidth=2)

        ax.set_xlabel('Iteración', fontsize=12)
        ax.set_ylabel(f'Tasa de Aceptación (ventana de {window_size})', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Líneas de referencia
        ax.axhline(0.4, color='green', linestyle='--', alpha=0.5,
                   label='Tasa ideal (~40%)')
        ax.legend()

    plt.tight_layout()
    return fig


def plot_multiple_runs(
    results: List[Dict],
    optimal_energy: Optional[float] = None,
    title: str = "Comparación de Múltiples Ejecuciones de SA",
    figsize: tuple = (12, 8)
) -> plt.Figure:
    """
    Visualiza múltiples ejecuciones de SA para evaluar variabilidad.

    Args:
        results: Lista de resultados de simulated_annealing()
        optimal_energy: Energía óptima conocida
        title: Título de la figura
        figsize: Tamaño de la figura

    Returns:
        Figure de matplotlib

    Examples:
        >>> multi_result = run_multiple_sa(grid, n_runs=10, max_iter=1000)
        >>> fig = plot_multiple_runs(multi_result['results'])
        >>> plt.show()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Panel izquierdo: Evolución de energías
    for i, result in enumerate(results):
        ax1.plot(result['energy_history'], alpha=0.5, linewidth=0.8,
                 label=f'Run {i + 1}')

    if optimal_energy is not None:
        ax1.axhline(optimal_energy, color='red', linestyle='--',
                    linewidth=2, label='Óptimo global')

    ax1.set_xlabel('Iteración', fontsize=12)
    ax1.set_ylabel('Energía', fontsize=12)
    ax1.set_title('Evolución de Energía (Múltiples Runs)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)

    # Panel derecho: Distribución de mejores energías
    best_energies = [result['best_energy'] for result in results]
    ax2.hist(best_energies, bins=15, alpha=0.7, color='#3498db',
             edgecolor='black')

    if optimal_energy is not None:
        ax2.axvline(optimal_energy, color='red', linestyle='--',
                    linewidth=2, label='Óptimo global')

    ax2.set_xlabel('Mejor Energía Encontrada', fontsize=12)
    ax2.set_ylabel('Frecuencia', fontsize=12)
    ax2.set_title('Distribución de Mejores Energías', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig
