"""
Funciones de visualización para el Problema 2.

Implementa todas las gráficas requeridas en el Punto 3:
1. GRÁFICA 1: Configuración óptima de la grilla 10×10
2. GRÁFICA 2: Energía vs iteración con fases identificadas
3. Comparación de múltiples runs
4. Métricas espaciales
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional, Tuple

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')


def plot_grid_configuration(
    grid_array: np.ndarray,
    Ti_positions: Optional[np.ndarray] = None,
    energia: Optional[float] = None,
    title: str = "Configuración Óptima - Grilla 10×10",
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    GRÁFICA 1: Visualiza la configuración de la grilla 10×10.

    Muestra los átomos de Fe, Nd y Ti con colores diferentes
    y destaca el núcleo central de Nd.

    Args:
        grid_array: Array (10, 10) con valores 0=Fe, 1=Nd, 2=Ti
        Ti_positions: Array (8, 2) opcional para destacar Ti
        energia: Energía de la configuración (opcional, para título)
        title: Título de la gráfica
        figsize: Tamaño de la figura
        save_path: Ruta para guardar (opcional)

    Returns:
        Figura de matplotlib

    Examples:
        >>> grid, Ti_pos, _ = crear_grid_inicial()
        >>> fig = plot_grid_configuration(grid, Ti_pos, energia=-123.45)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Colores para cada tipo de átomo
    colors = {
        0: '#87CEEB',  # Fe: azul claro
        1: '#FFD700',  # Nd: dorado
        2: '#FF6347'   # Ti: rojo tomate
    }

    # Dibujar grilla
    for i in range(10):
        for j in range(10):
            atom_type = grid_array[i, j]
            color = colors[atom_type]

            # Círculo para el átomo
            circle = plt.Circle((j, i), 0.35, color=color, ec='black', linewidth=1.5, zorder=2)
            ax.add_patch(circle)

            # Etiqueta
            labels = {0: 'Fe', 1: 'Nd', 2: 'Ti'}
            ax.text(j, i, labels[atom_type], ha='center', va='center',
                   fontsize=9, fontweight='bold', zorder=3)

    # Destacar núcleo de Nd con rectángulo
    rect = Rectangle((2.5, 2.5), 4, 4, fill=False, edgecolor='gold',
                     linewidth=3, linestyle='--', label='Núcleo Nd 4×4')
    ax.add_patch(rect)

    # Destacar posiciones de Ti si se proporciona
    if Ti_positions is not None:
        # IMPORTANTE: Ti_positions está en coordenadas físicas (Angstroms)
        # Convertir a índices de grilla para visualización
        GRID_SPACING = 2.8
        for idx in range(len(Ti_positions)):
            # Convertir de Angstroms a índices
            x_idx = int(np.round(Ti_positions[idx, 0] / GRID_SPACING))
            y_idx = int(np.round(Ti_positions[idx, 1] / GRID_SPACING))

            # Círculo exterior para Ti (usar índices para la grilla)
            circle_outer = plt.Circle((y_idx, x_idx), 0.45, fill=False, edgecolor='red',
                                     linewidth=2, linestyle=':', zorder=1)
            ax.add_patch(circle_outer)

    # Configurar ejes
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Para que (0,0) esté arriba a la izquierda

    # Grid
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Labels
    ax.set_xlabel('Columna (y)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fila (x)', fontsize=12, fontweight='bold')

    # Título
    if energia is not None:
        title += f"\nEnergía: {energia:.6f}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], edgecolor='black', label='Fe (Hierro)'),
        Patch(facecolor=colors[1], edgecolor='black', label='Nd (Neodimio)'),
        Patch(facecolor=colors[2], edgecolor='black', label='Ti (Titanio)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
             fontsize=10, frameon=True, shadow=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfica guardada en: {save_path}")

    return fig


def plot_energy_evolution(
    history: Dict[str, np.ndarray],
    highlight_phases: bool = True,
    exploration_threshold: float = 0.3,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    GRÁFICA 2: Energía vs iteración con fases identificadas.

    Visualiza la evolución de la energía durante el SA y marca las fases
    de exploración (T alta) y explotación (T baja).

    Args:
        history: Dict con claves 'energy', 'temperature', 'accepted'
        highlight_phases: Si True, marca fases de exploración/explotación
        exploration_threshold: Fracción de T0 para separar fases (default: 0.3)
        figsize: Tamaño de la figura
        save_path: Ruta para guardar (opcional)

    Returns:
        Figura de matplotlib

    Examples:
        >>> grid, Ti_pos, _ = crear_grid_inicial()
        >>> _, _, history = simulated_annealing(grid, Ti_pos, params, 20.0, 0.98, 10000)
        >>> fig = plot_energy_evolution(history, highlight_phases=True)
        >>> plt.show()
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    iterations = np.arange(len(history['energy']))
    energy = history['energy']
    temperature = history['temperature']
    T0 = temperature[0]

    # === SUBPLOT 1: Energía ===
    ax1.plot(iterations, energy, color='#2E86AB', linewidth=1, alpha=0.8, label='Energía')

    # Mejor energía encontrada
    best_energy = np.minimum.accumulate(energy)
    ax1.plot(iterations, best_energy, color='#A23B72', linewidth=2,
            label='Mejor Energía', linestyle='--')

    # Identificar fases si se solicita
    if highlight_phases:
        T_threshold = T0 * exploration_threshold

        # Encontrar donde termina exploración
        idx_threshold = np.where(temperature < T_threshold)[0]
        if len(idx_threshold) > 0:
            iter_threshold = idx_threshold[0]

            # Sombrear fases
            ax1.axvspan(0, iter_threshold, alpha=0.15, color='orange',
                       label=f'Exploración (T > {exploration_threshold}·T₀)')
            ax1.axvspan(iter_threshold, len(iterations), alpha=0.15, color='green',
                       label=f'Explotación (T ≤ {exploration_threshold}·T₀)')

            # Línea vertical separadora
            ax1.axvline(iter_threshold, color='red', linestyle=':', linewidth=2,
                       alpha=0.7, label=f'Transición (iter {iter_threshold})')

    ax1.set_ylabel('Energía', fontsize=12, fontweight='bold')
    ax1.set_title('Evolución de Energía durante Simulated Annealing',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    # === SUBPLOT 2: Temperatura ===
    ax2.plot(iterations, temperature, color='#F18F01', linewidth=1.5)
    ax2.set_ylabel('Temperatura', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Iteración', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')  # Escala logarítmica para temperatura
    ax2.grid(True, alpha=0.3, which='both')

    if highlight_phases and len(idx_threshold) > 0:
        ax2.axvline(iter_threshold, color='red', linestyle=':', linewidth=2, alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfica guardada en: {save_path}")

    return fig


def plot_acceptance_rate(
    history: Dict[str, np.ndarray],
    window_size: int = 500,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Grafica la tasa de aceptación vs iteración.

    Args:
        history: Dict con clave 'accepted' (booleano)
        window_size: Tamaño de ventana para calcular tasa (default: 500)
        figsize: Tamaño de la figura
        save_path: Ruta para guardar (opcional)

    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    accepted = history['accepted'].astype(float)
    iterations = np.arange(len(accepted))

    # Calcular tasa de aceptación con ventana deslizante
    acceptance_rate = np.convolve(accepted, np.ones(window_size)/window_size, mode='valid')
    iterations_rate = iterations[window_size-1:]

    ax.plot(iterations_rate, acceptance_rate, color='#06A77D', linewidth=2)
    ax.set_xlabel('Iteración', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Tasa de Aceptación (ventana {window_size})', fontsize=12, fontweight='bold')
    ax.set_title('Evolución de la Tasa de Aceptación', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_multiple_runs_comparison(
    resultados: List[Dict],
    top_n: int = 5,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
    auto_simplify: bool = True
) -> plt.Figure:
    """
    Compara múltiples runs de SA.

    Detecta automáticamente si hay baja variabilidad (σ ≈ 0) y simplifica
    la visualización para mostrar solo información relevante.

    Args:
        resultados: Lista de dicts con resultados de runs
        top_n: Número de mejores runs a mostrar
        figsize: Tamaño de la figura
        save_path: Ruta para guardar (opcional)
        auto_simplify: Si True, simplifica cuando detecta baja variabilidad

    Returns:
        Figura de matplotlib
    """
    # Calcular variabilidad
    energias_finales = [r['energia_final'] for r in resultados]
    std_energia = np.std(energias_finales)
    mean_energia = np.mean(energias_finales)
    coef_variacion = std_energia / abs(mean_energia) if abs(mean_energia) > 1e-10 else 0

    # Si hay MUY baja variabilidad (coef. variación < 0.01%), simplificar
    if auto_simplify and coef_variacion < 0.0001:
        # MODO SIMPLIFICADO: Solo gráfica de evolución + estadísticas
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]*0.6))

        # Ordenar por energía final
        resultados_sorted = sorted(resultados, key=lambda x: x['energia_final'])

        # Evolución de energía (top N runs)
        colors = plt.cm.viridis(np.linspace(0, 1, top_n))
        for idx, resultado in enumerate(resultados_sorted[:top_n]):
            if 'energy_history' in resultado:
                ax1.plot(resultado['energy_history'], color=colors[idx],
                        alpha=0.7, linewidth=1.5, label=f"Run {resultado['run_id']}")

        ax1.set_xlabel('Iteración', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Energía', fontsize=12, fontweight='bold')
        ax1.set_title(f'Evolución de Energía - Top {top_n} Runs\n' +
                     f'(Convergencia perfecta: σ={std_energia:.6f}, todos los {len(resultados)} runs → E={mean_energia:.6f})',
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9, loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Agregar nota
        mejoras = [r['mejora_relativa'] * 100 for r in resultados]
        iter_convergencias = [r['iterations_to_best'] for r in resultados]

        nota = (f"✓ Convergencia perfecta detectada\n"
                f"  • {len(resultados)} runs → misma energía\n"
                f"  • Mejora promedio: {np.mean(mejoras):.2f}%\n"
                f"  • Convergencia media: {np.mean(iter_convergencias):.0f} iter")

        ax1.text(0.02, 0.02, nota, transform=ax1.transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        plt.tight_layout()

        # Print estadísticas
        print("\n" + "="*70)
        print(f"CONVERGENCIA PERFECTA DETECTADA")
        print("="*70)
        print(f"Todos los {len(resultados)} runs convergieron a la MISMA energía")
        print(f"  → E_final = {mean_energia:.6f}")
        print(f"  → σ = {std_energia:.10f} (prácticamente 0)")
        print(f"  → Coef. variación = {coef_variacion*100:.6f}%")
        print("="*70)

    else:
        # MODO COMPLETO: 4 subplots como antes
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # Ordenar por energía final
        resultados_sorted = sorted(resultados, key=lambda x: x['energia_final'])

        # === SUBPLOT 1: Evolución de energía (top N runs) ===
        colors = plt.cm.viridis(np.linspace(0, 1, top_n))
        for idx, resultado in enumerate(resultados_sorted[:top_n]):
            if 'energy_history' in resultado:
                ax1.plot(resultado['energy_history'], color=colors[idx],
                        alpha=0.7, linewidth=1.5, label=f"Run {resultado['run_id']}")

        ax1.set_xlabel('Iteración', fontweight='bold')
        ax1.set_ylabel('Energía', fontweight='bold')
        ax1.set_title(f'Evolución de Energía - Top {top_n} Runs', fontweight='bold')
        ax1.legend(fontsize=8, loc='upper right')
        ax1.grid(True, alpha=0.3)

        # === SUBPLOT 2: Distribución de energías finales ===
        ax2.hist(energias_finales, bins=20, color='#2E86AB', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(energias_finales), color='red', linestyle='--',
                   linewidth=2, label=f'Media: {np.mean(energias_finales):.4f}')
        ax2.axvline(np.min(energias_finales), color='green', linestyle='--',
                   linewidth=2, label=f'Mejor: {np.min(energias_finales):.4f}')
        ax2.set_xlabel('Energía Final', fontweight='bold')
        ax2.set_ylabel('Frecuencia', fontweight='bold')
        ax2.set_title(f'Distribución de Energías Finales ({len(resultados)} runs)', fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')

        # === SUBPLOT 3: Mejora relativa ===
        mejoras = [r['mejora_relativa'] * 100 for r in resultados]
        run_ids = [r['run_id'] for r in resultados_sorted]
        ax3.bar(range(len(resultados_sorted)), [r['mejora_relativa']*100 for r in resultados_sorted],
               color='#06A77D', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Run ID', fontweight='bold')
        ax3.set_ylabel('Mejora (%)', fontweight='bold')
        ax3.set_title('Mejora Relativa por Run', fontweight='bold')
        ax3.set_xticks(range(0, len(resultados_sorted), max(1, len(resultados_sorted)//10)))
        ax3.grid(True, alpha=0.3, axis='y')

        # === SUBPLOT 4: Estadísticas ===
        ax4.axis('off')
        stats_text = f"""
        ESTADÍSTICAS DE {len(resultados)} RUNS
        {'='*40}

        Energía Final:
          - Mejor:    {np.min(energias_finales):.6f}
          - Peor:     {np.max(energias_finales):.6f}
          - Media:    {np.mean(energias_finales):.6f}
          - Std:      {np.std(energias_finales):.6f}

        Mejora:
          - Promedio: {np.mean(mejoras):.2f}%
          - Máxima:   {np.max(mejoras):.2f}%

        Convergencia:
          - Iter. promedio: {np.mean([r['iterations_to_best'] for r in resultados]):.0f}
        """
        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()

    # Guardar si se especifica (común para ambos modos)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_spatial_metrics(
    patron: Dict[str, float],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualiza las métricas espaciales del patrón óptimo.

    Args:
        patron: Dict retornado por analizar_patron_espacial()
        figsize: Tamaño de la figura
        save_path: Ruta para guardar (opcional)

    Returns:
        Figura de matplotlib
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    # Métricas a visualizar
    metricas_Ti_Nd = {
        'Promedio': patron['dist_Ti_Nd_promedio'],
        'Mínima': patron['dist_Ti_Nd_min'],
        'Máxima': patron['dist_Ti_Nd_max']
    }

    metricas_Ti_Ti = {
        'Promedio': patron['dist_Ti_Ti_promedio'],
        'Mínima': patron['dist_Ti_Ti_min'],
        'Máxima': patron['dist_Ti_Ti_max']
    }

    # === SUBPLOT 1: Distancias Ti-Nd ===
    ax1.bar(metricas_Ti_Nd.keys(), metricas_Ti_Nd.values(),
           color=['#2E86AB', '#A23B72', '#F18F01'], edgecolor='black')
    ax1.set_ylabel('Distancia (Å)', fontweight='bold')
    ax1.set_title('Distancias Ti-Nd', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # === SUBPLOT 2: Distancias Ti-Ti ===
    ax2.bar(metricas_Ti_Ti.keys(), metricas_Ti_Ti.values(),
           color=['#2E86AB', '#A23B72', '#F18F01'], edgecolor='black')
    ax2.set_ylabel('Distancia (Å)', fontweight='bold')
    ax2.set_title('Distancias Ti-Ti', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # === SUBPLOT 3: Clustering Score ===
    clustering = patron['clustering_score']
    colors_clustering = ['green' if clustering < 0.3 else 'orange' if clustering < 0.7 else 'red']
    ax3.barh(['Clustering Score'], [clustering], color=colors_clustering, edgecolor='black')
    ax3.set_xlim([0, 1])
    ax3.set_xlabel('Score (0=disperso, 1=agrupado)', fontweight='bold')
    ax3.set_title('Score de Agrupamiento', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    # === SUBPLOT 4: Resumen ===
    ax4.axis('off')
    resumen = f"""
    ANÁLISIS ESPACIAL
    {'='*30}

    Ti-Nd (¿se alejan?):
      Dist. promedio: {patron['dist_Ti_Nd_promedio']:.3f} Å
      {'→ SÍ se alejan' if patron['dist_Ti_Nd_promedio'] > 4.0 else '→ Relativamente cerca'}

    Ti-Ti (¿se dispersan?):
      Dist. promedio: {patron['dist_Ti_Ti_promedio']:.3f} Å
      Clustering: {patron['clustering_score']:.3f}
      {'→ DISPERSOS' if patron['clustering_score'] < 0.3 else '→ AGRUPADOS' if patron['clustering_score'] > 0.7 else '→ INTERMEDIOS'}

    Distribución radial:
      Dist. al centro: {patron['dist_centro_promedio']:.3f}
      {'→ PERIFERIA' if patron['dist_centro_promedio'] > 5.0 else '→ CENTRO'}
    """
    ax4.text(0.1, 0.5, resumen, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
