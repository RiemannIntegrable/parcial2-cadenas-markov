"""
Visualización de configuraciones 3D y resultados del Simulated Annealing.

Este módulo usa:
- Plotly para visualizaciones 3D interactivas
- Matplotlib para proyecciones 2D y gráficas de análisis
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from typing import Optional, Dict, List


def plot_configuration_3d_plotly(
    Fe_candidate_positions: np.ndarray,
    Nd_positions: np.ndarray,
    Ti_indices: np.ndarray,
    energia: float,
    title: str = "Configuración Cristalina 3D: NdFe₁₂"
) -> go.Figure:
    """
    Visualiza la configuración 3D interactiva con Plotly.

    Args:
        Fe_candidate_positions: Array (96, 3) con posiciones candidatas de Fe
        Nd_positions: Array (16, 3) con posiciones de Nd
        Ti_indices: Array (8,) con índices de átomos de Ti
        energia: Energía de la configuración
        title: Título de la figura

    Returns:
        go.Figure: Figura de Plotly interactiva

    Examples:
        >>> fig = plot_configuration_3d_plotly(Fe_pos, Nd_pos, Ti_idx, -456.78)
        >>> fig.show()
    """
    # Separar Fe y Ti
    Fe_mask = np.ones(96, dtype=bool)
    Fe_mask[Ti_indices] = False
    Fe_indices = np.where(Fe_mask)[0]

    Fe_positions = Fe_candidate_positions[Fe_indices]
    Ti_positions = Fe_candidate_positions[Ti_indices]

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
            size=4,
            color='lightblue',
            opacity=0.6,
            line=dict(color='navy', width=0.5)
        ),
        hovertemplate='<b>Fe</b><br>X: %{x:.3f} Å<br>Y: %{y:.3f} Å<br>Z: %{z:.3f} Å<extra></extra>'
    ))

    # Añadir átomos de Nd (dorado)
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

    # Añadir átomos de Ti (rojo, destacados)
    fig.add_trace(go.Scatter3d(
        x=Ti_positions[:, 0],
        y=Ti_positions[:, 1],
        z=Ti_positions[:, 2],
        mode='markers',
        name=f'Ti (n={len(Ti_positions)})',
        marker=dict(
            size=12,
            color='red',
            opacity=1.0,
            symbol='diamond',
            line=dict(color='darkred', width=3)
        ),
        hovertemplate='<b>Ti</b><br>X: %{x:.3f} Å<br>Y: %{y:.3f} Å<br>Z: %{z:.3f} Å<extra></extra>'
    ))

    # Configurar layout
    fig.update_layout(
        title=dict(
            text=f'{title}<br><sub>Energía: {energia:.6f}</sub>',
            font=dict(size=18, family='Arial')
        ),
        scene=dict(
            xaxis=dict(title='X (Å)', backgroundcolor='white', gridcolor='lightgray'),
            yaxis=dict(title='Y (Å)', backgroundcolor='white', gridcolor='lightgray'),
            zaxis=dict(title='Z (Å)', backgroundcolor='white', gridcolor='lightgray'),
            aspectmode='cube'
        ),
        width=1000,
        height=800,
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

    return fig


def plot_projections_2d(
    Fe_candidate_positions: np.ndarray,
    Nd_positions: np.ndarray,
    Ti_indices: np.ndarray,
    energia: float,
    figsize: tuple = (18, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualiza proyecciones 2D (XY, XZ, YZ) de la configuración 3D.

    Útil para comparar con Figura 29a de Skelland (proyección 2D).

    Args:
        Fe_candidate_positions: Array (96, 3) con posiciones candidatas de Fe
        Nd_positions: Array (16, 3) con posiciones de Nd
        Ti_indices: Array (8,) con índices de átomos de Ti
        energia: Energía de la configuración
        figsize: Tamaño de la figura
        save_path: Path para guardar figura (opcional)

    Returns:
        plt.Figure: Figura de matplotlib

    Examples:
        >>> fig = plot_projections_2d(Fe_pos, Nd_pos, Ti_idx, -456.78)
        >>> plt.show()
    """
    # Separar Fe y Ti
    Fe_mask = np.ones(96, dtype=bool)
    Fe_mask[Ti_indices] = False
    Fe_indices = np.where(Fe_mask)[0]

    Fe_positions = Fe_candidate_positions[Fe_indices]
    Ti_positions = Fe_candidate_positions[Ti_indices]

    # Crear figura con 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # Proyección XY
    ax1.scatter(Fe_positions[:, 0], Fe_positions[:, 1], c='lightblue',
                s=30, alpha=0.6, label='Fe', edgecolors='navy', linewidth=0.5)
    ax1.scatter(Nd_positions[:, 0], Nd_positions[:, 1], c='gold',
                s=120, alpha=0.95, label='Nd', edgecolors='orange', linewidth=2, marker='s')
    ax1.scatter(Ti_positions[:, 0], Ti_positions[:, 1], c='red',
                s=200, alpha=1.0, label='Ti', edgecolors='darkred', linewidth=3, marker='D')
    ax1.set_xlabel('X (Å)', fontsize=12)
    ax1.set_ylabel('Y (Å)', fontsize=12)
    ax1.set_title('Proyección XY', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Proyección XZ
    ax2.scatter(Fe_positions[:, 0], Fe_positions[:, 2], c='lightblue',
                s=30, alpha=0.6, label='Fe', edgecolors='navy', linewidth=0.5)
    ax2.scatter(Nd_positions[:, 0], Nd_positions[:, 2], c='gold',
                s=120, alpha=0.95, label='Nd', edgecolors='orange', linewidth=2, marker='s')
    ax2.scatter(Ti_positions[:, 0], Ti_positions[:, 2], c='red',
                s=200, alpha=1.0, label='Ti', edgecolors='darkred', linewidth=3, marker='D')
    ax2.set_xlabel('X (Å)', fontsize=12)
    ax2.set_ylabel('Z (Å)', fontsize=12)
    ax2.set_title('Proyección XZ', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Proyección YZ
    ax3.scatter(Fe_positions[:, 1], Fe_positions[:, 2], c='lightblue',
                s=30, alpha=0.6, label='Fe', edgecolors='navy', linewidth=0.5)
    ax3.scatter(Nd_positions[:, 1], Nd_positions[:, 2], c='gold',
                s=120, alpha=0.95, label='Nd', edgecolors='orange', linewidth=2, marker='s')
    ax3.scatter(Ti_positions[:, 1], Ti_positions[:, 2], c='red',
                s=200, alpha=1.0, label='Ti', edgecolors='darkred', linewidth=3, marker='D')
    ax3.set_xlabel('Y (Å)', fontsize=12)
    ax3.set_ylabel('Z (Å)', fontsize=12)
    ax3.set_title('Proyección YZ', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    fig.suptitle(f'Proyecciones 2D de Configuración Óptima | Energía: {energia:.6f}',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figura guardada en: {save_path}")

    return fig


def plot_energy_evolution(
    energy_history: np.ndarray,
    title: str = "Evolución de Energía - Simulated Annealing 3D",
    highlight_phases: bool = True,
    exploration_threshold: float = 0.5,
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualiza la evolución de energía durante el SA.

    Args:
        energy_history: Array con historia de energías
        title: Título de la figura
        highlight_phases: Si True, resalta fases de exploración/explotación
        exploration_threshold: Umbral para detectar fase de exploración
        figsize: Tamaño de la figura
        save_path: Path para guardar figura (opcional)

    Returns:
        plt.Figure: Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    iterations = np.arange(len(energy_history))

    # Graficar energía
    ax.plot(iterations, energy_history, linewidth=1.5, alpha=0.8, color='steelblue')

    # Resaltar mejor energía encontrada
    best_energy_idx = np.argmin(energy_history)
    best_energy = energy_history[best_energy_idx]
    ax.axhline(best_energy, color='red', linestyle='--', linewidth=2,
               label=f'Mejor energía: {best_energy:.6f} (iter {best_energy_idx:,})', alpha=0.7)

    # Marcar punto de mejor energía
    ax.plot(best_energy_idx, best_energy, 'r*', markersize=20, label='Óptimo encontrado')

    ax.set_xlabel('Iteración', fontsize=12)
    ax.set_ylabel('Energía', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figura guardada en: {save_path}")

    return fig


def plot_spatial_metrics_3d(
    patron: Dict[str, float],
    figsize: tuple = (16, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualiza métricas espaciales 3D (distancias Ti-Nd, Ti-Ti, clustering).

    Args:
        patron: Dict retornado por analizar_patron_espacial_3d()
        figsize: Tamaño de la figura
        save_path: Path para guardar figura (opcional)

    Returns:
        plt.Figure: Figura de matplotlib
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Distancias Ti-Nd
    metrics_Ti_Nd = [
        patron['dist_Ti_Nd_promedio'],
        patron['dist_Ti_Nd_min'],
        patron['dist_Ti_Nd_max']
    ]
    labels_Ti_Nd = ['Promedio', 'Mínima', 'Máxima']
    colors_Ti_Nd = ['steelblue', 'green', 'red']

    ax1.bar(labels_Ti_Nd, metrics_Ti_Nd, color=colors_Ti_Nd, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Distancia (Å)', fontsize=12)
    ax1.set_title('Distancias Ti-Nd\n(Métrica de Skelland)', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(metrics_Ti_Nd):
        ax1.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')

    # Panel 2: Distancias Ti-Ti
    metrics_Ti_Ti = [
        patron['dist_Ti_Ti_promedio'],
        patron['dist_Ti_Ti_min'],
        patron['dist_Ti_Ti_max'],
        patron['dist_Ti_vecino_promedio']
    ]
    labels_Ti_Ti = ['Promedio\n(todos pares)', 'Mínima', 'Máxima', 'Vecino\nmás cercano']
    colors_Ti_Ti = ['steelblue', 'orange', 'red', 'purple']

    ax2.bar(labels_Ti_Ti, metrics_Ti_Ti, color=colors_Ti_Ti, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Distancia (Å)', fontsize=12)
    ax2.set_title('Distancias Ti-Ti\n(Métrica de Skelland)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate(metrics_Ti_Ti):
        ax2.text(i, v + 0.2, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')

    # Panel 3: Clustering score
    clustering_score = patron['clustering_score']
    ax3.barh(['Clustering Score'], [clustering_score], color='teal', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_xlim([0, 1])
    ax3.set_xlabel('Score (0=disperso, 1=agrupado)', fontsize=12)
    ax3.set_title('Análisis de Clustering', fontsize=13, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    ax3.text(clustering_score + 0.05, 0, f'{clustering_score:.3f}', va='center', fontsize=11, fontweight='bold')

    # Agregar interpretación
    if clustering_score < 0.3:
        interpretation = "Disperso ✓"
        color = 'green'
    elif clustering_score > 0.7:
        interpretation = "Agrupado"
        color = 'red'
    else:
        interpretation = "Intermedio"
        color = 'orange'
    ax3.text(0.5, -0.3, interpretation, ha='center', fontsize=12, fontweight='bold', color=color,
             transform=ax3.transAxes)

    fig.suptitle('Análisis Espacial 3D de Configuración Óptima', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figura guardada en: {save_path}")

    return fig


def plot_multiple_runs_comparison_3d(
    resultados: List[Dict],
    top_n: int = 10,
    figsize: tuple = (16, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compara múltiples runs de SA 3D.

    Args:
        resultados: Lista de dicts retornados por ejecutar_multiples_runs_3d()
        top_n: Número de mejores runs a resaltar
        figsize: Tamaño de la figura
        save_path: Path para guardar figura (opcional)

    Returns:
        plt.Figure: Figura de matplotlib
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    n_runs = len(resultados)
    run_ids = [r['run_id'] for r in resultados]
    energias_finales = [r['energia_final'] for r in resultados]
    mejoras = [r['mejora_relativa'] * 100 for r in resultados]

    # Ordenar por energía final
    sorted_indices = np.argsort(energias_finales)
    top_indices = sorted_indices[:top_n]

    # Panel 1: Distribución de energías finales
    ax1.hist(energias_finales, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(energias_finales), color='red', linestyle='--',
                linewidth=2, label=f'Media: {np.mean(energias_finales):.4f}')
    ax1.axvline(np.min(energias_finales), color='green', linestyle='--',
                linewidth=2, label=f'Mejor: {np.min(energias_finales):.4f}')
    ax1.set_xlabel('Energía Final', fontsize=11)
    ax1.set_ylabel('Frecuencia', fontsize=11)
    ax1.set_title(f'Distribución de Energías Finales ({n_runs} runs)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Scatter energía vs run_id
    colors = ['red' if i in top_indices else 'steelblue' for i in range(n_runs)]
    ax2.scatter(run_ids, energias_finales, c=colors, s=50, alpha=0.7, edgecolors='black')
    ax2.set_xlabel('Run ID', fontsize=11)
    ax2.set_ylabel('Energía Final', fontsize=11)
    ax2.set_title(f'Energías por Run (Top {top_n} en rojo)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Distribución de mejoras
    ax3.hist(mejoras, bins=30, color='teal', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(mejoras), color='red', linestyle='--',
                linewidth=2, label=f'Media: {np.mean(mejoras):.2f}%')
    ax3.set_xlabel('Mejora Relativa (%)', fontsize=11)
    ax3.set_ylabel('Frecuencia', fontsize=11)
    ax3.set_title('Distribución de Mejoras Relativas', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Evolución de energía del mejor run
    best_run = resultados[sorted_indices[0]]
    iterations = np.arange(len(best_run['energy_history']))
    ax4.plot(iterations, best_run['energy_history'], linewidth=1.5, alpha=0.8, color='steelblue')
    ax4.axhline(best_run['energia_final'], color='red', linestyle='--',
                linewidth=2, label=f'Mejor: {best_run["energia_final"]:.6f}', alpha=0.7)
    ax4.set_xlabel('Iteración', fontsize=11)
    ax4.set_ylabel('Energía', fontsize=11)
    ax4.set_title(f'Evolución del Mejor Run (ID {best_run["run_id"]})', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'Comparación de {n_runs} Runs de Simulated Annealing 3D',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figura guardada en: {save_path}")

    return fig
