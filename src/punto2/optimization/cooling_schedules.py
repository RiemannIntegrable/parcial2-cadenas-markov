"""
Esquemas de enfriamiento para Simulated Annealing.

Este módulo implementa diferentes estrategias de temperatura decreciente,
cada una con sus propias garantías teóricas y características prácticas.
"""

import numpy as np
from typing import Callable


def geometric_cooling(T0: float, alpha: float) -> Callable[[int], float]:
    """
    Enfriamiento geométrico: T(t) = T₀ · α^t

    Este es el esquema más común en la práctica. Es más agresivo que el
    logarítmico pero converge más rápido.

    Args:
        T0: Temperatura inicial
        alpha: Factor de enfriamiento (típicamente 0.8 ≤ α ≤ 0.99)

    Returns:
        Función que mapea iteración → temperatura

    Examples:
        >>> cooling = geometric_cooling(T0=10.0, alpha=0.95)
        >>> cooling(0)
        10.0
        >>> cooling(100) < cooling(50) < cooling(0)
        True

    Note:
        - α cercano a 1 → enfriamiento lento (más exploración)
        - α cercano a 0 → enfriamiento rápido (más explotación)
        - Típico: α ∈ [0.90, 0.99]
    """
    def temperature(iteration: int) -> float:
        return T0 * (alpha ** iteration)

    return temperature


def exponential_cooling(T0: float, beta: float) -> Callable[[int], float]:
    """
    Enfriamiento exponencial: T(t) = T₀ / (1 + β·t)

    Más suave que el geométrico al inicio, más agresivo al final.

    Args:
        T0: Temperatura inicial
        beta: Tasa de enfriamiento (típicamente 0.001 ≤ β ≤ 0.1)

    Returns:
        Función que mapea iteración → temperatura

    Examples:
        >>> cooling = exponential_cooling(T0=10.0, beta=0.01)
        >>> cooling(0)
        10.0
        >>> cooling(1000) < 1.0
        True
    """
    def temperature(iteration: int) -> float:
        return T0 / (1.0 + beta * iteration)

    return temperature


def logarithmic_cooling(c: float, t0: int = 1) -> Callable[[int], float]:
    """
    Enfriamiento logarítmico: T(t) = c / log(t + t₀)

    Garantiza convergencia al óptimo global (Teorema de Hajek),
    pero es EXTREMADAMENTE lento para aplicaciones prácticas.

    Args:
        c: Constante (debe ser ≥ altura de barreras de energía)
        t0: Offset para evitar log(0)

    Returns:
        Función que mapea iteración → temperatura

    Examples:
        >>> cooling = logarithmic_cooling(c=10.0)
        >>> cooling(1)  # log(2) ≈ 0.693
        14.426950408889634

    Warning:
        Solo usar para problemas MUY pequeños o como baseline teórico.
        Requiere millones de iteraciones para converger.
    """
    def temperature(iteration: int) -> float:
        return c / np.log(iteration + t0)

    return temperature


def adaptive_cooling(
    T0: float,
    alpha_fast: float = 0.85,
    alpha_slow: float = 0.98,
    target_acceptance_rate: float = 0.4
) -> Callable[[int, float], float]:
    """
    Enfriamiento adaptativo basado en tasa de aceptación.

    Ajusta la velocidad de enfriamiento según qué tan bien está explorando
    el algoritmo:
        - Tasa de aceptación alta → enfriamiento más rápido
        - Tasa de aceptación baja → enfriamiento más lento

    Args:
        T0: Temperatura inicial
        alpha_fast: Factor de enfriamiento cuando se acepta mucho
        alpha_slow: Factor de enfriamiento cuando se acepta poco
        target_acceptance_rate: Tasa objetivo (típicamente 0.3-0.5)

    Returns:
        Función que mapea (iteración, tasa_aceptación) → temperatura

    Examples:
        >>> cooling = adaptive_cooling(T0=10.0)
        >>> T1 = cooling(10, 0.8)  # Alta aceptación
        >>> T2 = cooling(10, 0.2)  # Baja aceptación
        >>> T1 < T2  # Enfría más rápido si acepta mucho
        True

    Note:
        Requiere trackear la tasa de aceptación durante la ejecución.
    """
    def temperature(iteration: int, acceptance_rate: float) -> float:
        # Interpolar alpha según tasa de aceptación
        if acceptance_rate > target_acceptance_rate:
            alpha = alpha_fast
        else:
            alpha = alpha_slow

        return T0 * (alpha ** iteration)

    return temperature


def linear_cooling(T0: float, Tf: float, max_iter: int) -> Callable[[int], float]:
    """
    Enfriamiento lineal: T(t) = T₀ - (T₀ - Tf) · (t / max_iter)

    Decremento constante de temperatura. Muy simple pero poco efectivo.

    Args:
        T0: Temperatura inicial
        Tf: Temperatura final
        max_iter: Número máximo de iteraciones

    Returns:
        Función que mapea iteración → temperatura

    Examples:
        >>> cooling = linear_cooling(T0=10.0, Tf=0.01, max_iter=1000)
        >>> np.isclose(cooling(0), 10.0)
        True
        >>> np.isclose(cooling(1000), 0.01)
        True
    """
    def temperature(iteration: int) -> float:
        return T0 - (T0 - Tf) * (iteration / max_iter)

    return temperature


def get_cooling_schedule(
    name: str,
    T0: float = 1.0,
    **kwargs
) -> Callable:
    """
    Factory function para obtener un esquema de enfriamiento por nombre.

    Args:
        name: Nombre del esquema ('geometric', 'exponential', 'logarithmic', 'linear')
        T0: Temperatura inicial
        **kwargs: Parámetros específicos del esquema

    Returns:
        Función de enfriamiento

    Raises:
        ValueError: Si el nombre no es reconocido

    Examples:
        >>> cooling = get_cooling_schedule('geometric', T0=10.0, alpha=0.95)
        >>> cooling(0)
        10.0
    """
    schedules = {
        'geometric': lambda: geometric_cooling(T0, kwargs.get('alpha', 0.95)),
        'exponential': lambda: exponential_cooling(T0, kwargs.get('beta', 0.01)),
        'logarithmic': lambda: logarithmic_cooling(kwargs.get('c', 10.0)),
        'linear': lambda: linear_cooling(T0, kwargs.get('Tf', 0.01), kwargs.get('max_iter', 1000))
    }

    if name not in schedules:
        raise ValueError(
            f"Esquema '{name}' no reconocido. "
            f"Opciones: {list(schedules.keys())}"
        )

    return schedules[name]()
