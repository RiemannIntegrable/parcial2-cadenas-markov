"""
Funciones de visualización para configuraciones 3D usando Plotly.

Este módulo proporciona funciones para visualizar la estructura cristalina 3D
y los resultados del algoritmo de optimización.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict


def plot_crystal_configuration_3d(
    all_positions: np.ndarray,
    atom_types: np.ndarray,
    Nd_positions: np.ndarray,
    Ti_indices: Optional[np.ndarray] = None,
    energia: Optional[float] = None,
    title: str = "Configuración Cristalina 3D",
    show_bonds: bool = False
):
    """
    Visualiza una configuración de la estructura cristalina en 3D.

    Args:
        all_positions: Array (112, 3) con coordenadas de todos los átomos
        atom_types: Array (112,) con tipos (0=Fe, 1=Nd, 2=Ti)
        Nd_positions: Array (16, 3) con posiciones de Nd (para referencia)
        Ti_indices: Array (8,) con índices de Ti (opcional)
        energia: Energía de la configuración (opcional)
        title: Título de la gráfica
        show_bonds: Si True, muestra enlaces entre átomos cercanos

    Returns:
        Figura de Plotly
    """
    fig = go.Figure()

    # Extraer posiciones por tipo
    Fe_mask = atom_types == 0
    Nd_mask = atom_types == 1
    Ti_mask = atom_types == 2

    Fe_pos = all_positions[Fe_mask]
    Nd_pos = all_positions[Nd_mask]
    Ti_pos = all_positions[Ti_mask]

    # Átomos de Fe (azul claro, pequeños)
    if len(Fe_pos) > 0:
        fig.add_trace(go.Scatter3d(
            x=Fe_pos[:, 0],
            y=Fe_pos[:, 1],
            z=Fe_pos[:, 2],
            mode='markers',
            name=f'Fe (n={len(Fe_pos)})',
            marker=dict(
                size=4,
                color='lightblue',
                opacity=0.5,
                line=dict(color='navy', width=0.5)
            ),
            hovertemplate='<b>Fe</b><br>X: %{x:.3f} Å<br>Y: %{y:.3f} Å<br>Z: %{z:.3f} Å<extra></extra>'
        ))

    # Átomos de Nd (amarillo, medianos)
    if len(Nd_pos) > 0:
        fig.add_trace(go.Scatter3d(
            x=Nd_pos[:, 0],
            y=Nd_pos[:, 1],
            z=Nd_pos[:, 2],
            mode='markers',
            name=f'Nd (n={len(Nd_pos)})',
            marker=dict(
                size=10,
                color='gold',
                opacity=0.95,
                line=dict(color='orange', width=1.5)
            ),
            hovertemplate='<b>Nd</b><br>X: %{x:.3f} Å<br>Y: %{y:.3f} Å<br>Z: %{z:.3f} Å<extra></extra>'
        ))

    # Átomos de Ti (rojo, grandes)
    if len(Ti_pos) > 0:
        fig.add_trace(go.Scatter3d(
            x=Ti_pos[:, 0],
            y=Ti_pos[:, 1],
            z=Ti_pos[:, 2],
            mode='markers',
            name=f'Ti (n={len(Ti_pos)})',
            marker=dict(
                size=8,
                color='red',
                opacity=1.0,
                line=dict(color='darkred', width=2)
            ),
            hovertemplate='<b>Ti</b><br>X: %{x:.3f} Å<br>Y: %{y:.3f} Å<br>Z: %{z:.3f} Å<extra></extra>'
        ))

    # Título con energía si se proporciona
    if energia is not None:
        title = f"{title}<br>Energía: {energia:.6f}"

    # Configurar layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family='Arial Black')),
        scene=dict(
            xaxis=dict(title='X (Å)', backgroundcolor='white', gridcolor='lightgray'),
            yaxis=dict(title='Y (Å)', backgroundcolor='white', gridcolor='lightgray'),
            zaxis=dict(title='Z (Å)', backgroundcolor='white', gridcolor='lightgray'),
            aspectmode='data'
        ),
        width=900,
        height=700,
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


def plot_energy_evolution_3d(
    history: Dict,
    highlight_convergence: bool = True,
    figsize: tuple = (14, 6)
):
    """
    Visualiza la evolución de la energía durante el algoritmo.

    Args:
        history: Dict con 'energy', 'temperature' (submuestreados)
        highlight_convergence: Si True, marca el punto de convergencia
        figsize: Tamaño de la figura

    Returns:
        Figura de Plotly
    """
    from plotly.subplots import make_subplots

    energy = history['energy']
    temperature = history.get('temperature', None)

    # Crear subplots
    if temperature is not None:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Evolución de Energía', 'Temperatura'),
            horizontal_spacing=0.1
        )
    else:
        fig = go.Figure()

    # Plot de energía
    fig.add_trace(
        go.Scatter(
            y=energy,
            mode='lines',
            name='Energía',
            line=dict(color='blue', width=1.5)
        ),
        row=1, col=1 if temperature is not None else None
    )

    # Marcar convergencia si se proporciona
    if highlight_convergence and 'iterations_to_best' in history:
        iter_best = history['iterations_to_best']
        # Convertir a índice submuestreado (asumiendo save_every=10)
        iter_best_idx = iter_best // 10
        if iter_best_idx < len(energy):
            fig.add_trace(
                go.Scatter(
                    x=[iter_best_idx],
                    y=[energy[iter_best_idx]],
                    mode='markers',
                    name='Óptimo',
                    marker=dict(size=12, color='red', symbol='star')
                ),
                row=1, col=1 if temperature is not None else None
            )

    # Plot de temperatura
    if temperature is not None:
        fig.add_trace(
            go.Scatter(
                y=temperature,
                mode='lines',
                name='Temperatura',
                line=dict(color='orange', width=1.5)
            ),
            row=1, col=2
        )

    # Layout
    fig.update_layout(
        height=400,
        width=figsize[0] * 70,
        showlegend=True,
        title_text="Evolución del Algoritmo de Simulated Annealing"
    )

    fig.update_xaxes(title_text="Iteración (submuestreado)", row=1, col=1)
    fig.update_yaxes(title_text="Energía", row=1, col=1)

    if temperature is not None:
        fig.update_xaxes(title_text="Iteración (submuestreado)", row=1, col=2)
        fig.update_yaxes(title_text="Temperatura", row=1, col=2)

    return fig


def plot_multiple_runs_comparison_3d(
    resultados: List[Dict],
    top_n: int = 10
):
    """
    Compara los resultados de múltiples runs.

    Args:
        resultados: Lista de diccionarios con resultados de runs
        top_n: Número de mejores runs a mostrar

    Returns:
        Figura de Plotly
    """
    # Ordenar por energía final
    resultados_sorted = sorted(resultados, key=lambda x: x['energia_final'])[:top_n]

    run_ids = [r['run_id'] for r in resultados_sorted]
    energias = [r['energia_final'] for r in resultados_sorted]
    mejoras = [r['mejora_relativa'] * 100 for r in resultados_sorted]

    # Crear subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'Top {top_n} Energías Finales', f'Top {top_n} Mejoras Relativas'),
        horizontal_spacing=0.15
    )

    # Plot de energías
    fig.add_trace(
        go.Bar(
            x=run_ids,
            y=energias,
            name='Energía Final',
            marker=dict(color='steelblue')
        ),
        row=1, col=1
    )

    # Plot de mejoras
    fig.add_trace(
        go.Bar(
            x=run_ids,
            y=mejoras,
            name='Mejora (%)',
            marker=dict(color='seagreen')
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=400,
        width=1000,
        showlegend=False,
        title_text=f"Comparación de {len(resultados)} Runs Independientes"
    )

    fig.update_xaxes(title_text="Run ID", row=1, col=1)
    fig.update_yaxes(title_text="Energía", row=1, col=1)

    fig.update_xaxes(title_text="Run ID", row=1, col=2)
    fig.update_yaxes(title_text="Mejora (%)", row=1, col=2)

    return fig
