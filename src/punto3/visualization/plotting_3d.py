"""
Visualizaciones 3D interactivas con Plotly.

Este módulo provee funciones para crear gráficas 3D interactivas de la estructura
cristalina usando Plotly, permitiendo rotación, zoom y hover interactivo.
"""

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from typing import Optional, Dict


def plot_crystal_configuration_3d(
    all_positions: np.ndarray,
    atom_types: np.ndarray,
    energia: Optional[float] = None,
    title: str = "Configuración Cristalina 3D",
    figsize: tuple = (1000, 800),
    show_fig: bool = True
) -> go.Figure:
    """
    Crea una gráfica 3D interactiva de la configuración cristalina.

    Args:
        all_positions: Array (N, 3) con posiciones de todos los átomos
        atom_types: Array (N,) con tipos [0=Fe, 1=Nd, 2=Ti]
        energia: Energía total del sistema (opcional, se muestra en título)
        title: Título de la gráfica
        figsize: Tupla (width, height) en píxeles
        show_fig: Si True, muestra la figura inmediatamente

    Returns:
        Figura de Plotly (puede guardarse como HTML o imagen)

    Example:
        >>> fig = plot_crystal_configuration_3d(all_pos, types, energia=-12345.67)
        >>> fig.write_html("configuracion.html")
    """
    # Separar átomos por tipo
    Fe_mask = (atom_types == 0)
    Nd_mask = (atom_types == 1)
    Ti_mask = (atom_types == 2)

    Fe_positions = all_positions[Fe_mask]
    Nd_positions = all_positions[Nd_mask]
    Ti_positions = all_positions[Ti_mask]

    # Crear figura
    fig = go.Figure()

    # Añadir átomos de Fe (azul claro)
    fig.add_trace(go.Scatter3d(
        x=Fe_positions[:, 0],
        y=Fe_positions[:, 1],
        z=Fe_positions[:, 2],
        mode='markers',
        name=f'Fe (n={len(Fe_positions)})',
        marker=dict(
            size=5,
            color='lightblue',
            opacity=0.6,
            line=dict(color='navy', width=0.5)
        ),
        hovertemplate='<b>Fe</b><br>X: %{x:.3f} Å<br>Y: %{y:.3f} Å<br>Z: %{z:.3f} Å<extra></extra>'
    ))

    # Añadir átomos de Nd (amarillo/dorado)
    fig.add_trace(go.Scatter3d(
        x=Nd_positions[:, 0],
        y=Nd_positions[:, 1],
        z=Nd_positions[:, 2],
        mode='markers',
        name=f'Nd (n={len(Nd_positions)})',
        marker=dict(
            size=10,
            color='gold',
            opacity=0.95,
            line=dict(color='orange', width=2)
        ),
        hovertemplate='<b>Nd</b><br>X: %{x:.3f} Å<br>Y: %{y:.3f} Å<br>Z: %{z:.3f} Å<extra></extra>'
    ))

    # Añadir átomos de Ti (rojo)
    if len(Ti_positions) > 0:
        fig.add_trace(go.Scatter3d(
            x=Ti_positions[:, 0],
            y=Ti_positions[:, 1],
            z=Ti_positions[:, 2],
            mode='markers',
            name=f'Ti (n={len(Ti_positions)})',
            marker=dict(
                size=8,
                color='red',
                opacity=0.9,
                line=dict(color='darkred', width=2)
            ),
            hovertemplate='<b>Ti</b><br>X: %{x:.3f} Å<br>Y: %{y:.3f} Å<br>Z: %{z:.3f} Å<extra></extra>'
        ))

    # Título con energía si se proporciona
    if energia is not None:
        title_text = f"{title}<br><sub>Energía Total: {energia:.4f} eV</sub>"
    else:
        title_text = title

    # Configurar layout
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=18, family='Arial Black')
        ),
        scene=dict(
            xaxis=dict(
                title='X (Å)',
                backgroundcolor='white',
                gridcolor='lightgray',
                showbackground=True
            ),
            yaxis=dict(
                title='Y (Å)',
                backgroundcolor='white',
                gridcolor='lightgray',
                showbackground=True
            ),
            zaxis=dict(
                title='Z (Å)',
                backgroundcolor='white',
                gridcolor='lightgray',
                showbackground=True
            ),
            aspectmode='cube'
        ),
        width=figsize[0],
        height=figsize[1],
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        hovermode='closest'
    )

    if show_fig:
        fig.show()

    return fig


def plot_energy_evolution_3d(
    history: Dict,
    title: str = "Evolución de Energía - Simulated Annealing 3D",
    figsize: tuple = (16, 7)
) -> plt.Figure:
    """
    Crea gráfica de evolución de energía y temperatura vs iteración.

    Esta función usa matplotlib (no Plotly) porque es más apropiada para
    gráficas de series temporales con múltiples subplots.

    Args:
        history: Diccionario con 'energy', 'temperature' (y opcionalmente 'accepted')
        title: Título de la gráfica
        figsize: Tupla (width, height) en pulgadas

    Returns:
        Figura de matplotlib
    """
    energy = history['energy']
    temperature = history.get('temperature', None)

    n_iter = len(energy)
    iterations = np.arange(n_iter)

    # Crear figura con 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Subplot 1: Energía
    ax1.plot(iterations, energy, 'b-', linewidth=1, alpha=0.7, label='Energía')
    ax1.axhline(np.min(energy), color='red', linestyle='--', linewidth=2,
                label=f'Óptimo: {np.min(energy):.4f} eV')
    ax1.set_ylabel('Energía (eV)', fontsize=12, fontweight='bold')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Temperatura
    if temperature is not None:
        ax2.plot(iterations, temperature, 'r-', linewidth=1.5, label='Temperatura')
        ax2.set_ylabel('Temperatura', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Iteración', fontsize=12, fontweight='bold')
        ax2.set_yscale('log')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_spatial_metrics_3d(
    patron: Dict,
    figsize: tuple = (14, 8)
) -> plt.Figure:
    """
    Crea histogramas de métricas espaciales.

    Args:
        patron: Diccionario retornado por analizar_patron_espacial_3d
        figsize: Tupla (width, height) en pulgadas

    Returns:
        Figura de matplotlib con histogramas
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Subplot 1: Distancias Ti-Nd
    ax1 = axes[0, 0]
    ax1.hist(patron['dist_Ti_Nd_all'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(patron['dist_Ti_Nd_promedio'], color='red', linestyle='--', linewidth=2,
                label=f'Promedio: {patron["dist_Ti_Nd_promedio"]:.3f} Å')
    ax1.set_xlabel('Distancia Ti-Nd (Å)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    ax1.set_title('Distribución de Distancias Ti-Nd', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Distancias Ti-Ti
    ax2 = axes[0, 1]
    ax2.hist(patron['dist_Ti_Ti_all'], bins=15, color='lightcoral', edgecolor='black', alpha=0.7)
    ax2.axvline(patron['dist_Ti_Ti_promedio'], color='red', linestyle='--', linewidth=2,
                label=f'Promedio: {patron["dist_Ti_Ti_promedio"]:.3f} Å')
    ax2.set_xlabel('Distancia Ti-Ti (Å)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    ax2.set_title('Distribución de Distancias Ti-Ti', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Métricas resumidas
    ax3 = axes[1, 0]
    metrics_names = ['Dist Ti-Nd\n(Å)', 'Dist Ti-Ti\n(Å)', 'Clustering\nScore', 'Dist Centro\n(Å)']
    metrics_values = [
        patron['dist_Ti_Nd_promedio'],
        patron['dist_Ti_Ti_promedio'],
        patron['clustering_score'],
        patron['dist_centro_promedio']
    ]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']

    bars = ax3.bar(metrics_names, metrics_values, color=colors, edgecolor='black', alpha=0.7)
    ax3.set_ylabel('Valor', fontsize=11, fontweight='bold')
    ax3.set_title('Resumen de Métricas Espaciales', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Añadir valores sobre las barras
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    # Subplot 4: Texto con interpretación
    ax4 = axes[1, 1]
    ax4.axis('off')

    from ..analysis import interpretar_patron_3d
    interpretacion = interpretar_patron_3d(patron)

    ax4.text(0.1, 0.9, 'Interpretación:', fontsize=13, fontweight='bold',
             verticalalignment='top', transform=ax4.transAxes)
    ax4.text(0.1, 0.75, interpretacion, fontsize=10,
             verticalalignment='top', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    return fig
