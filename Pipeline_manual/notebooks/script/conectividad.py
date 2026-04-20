"""
conectividad.py
---------------
Métodos de análisis de conectividad funcional sobre señales ROI.

Métodos disponibles:
  - correlacion          : correlación de Pearson (numpy)
  - correlacion_parcial  : correlación parcial (Nilearn)
  - graphical_lasso      : covarianza inversa / precisión (Nilearn)
  - comparar_matrices    : resumen estadístico de ambas matrices
  - umbralizar           : pone en cero entradas menores al umbral
"""

import numpy as np
from nilearn.connectome import ConnectivityMeasure


# ─────────────────────────────────────────────────────────────────────────────
# Métodos de conectividad
# ─────────────────────────────────────────────────────────────────────────────

def correlacion(roi_signals_filt):
    """
    Calcula la matriz de correlación de Pearson entre ROIs.

    Cada entrada (i, j) indica cuán similares son las señales temporales
    de la ROI i y la ROI j. No distingue conexión directa de indirecta
    (una tercera ROI puede inflar la correlación entre dos).

    Parámetros
    ----------
    roi_signals_filt : array (T, n_rois) — señales filtradas (y opcionalmente z-score)

    Retorna
    -------
    corr_matrix : array (n_rois, n_rois) — valores en [-1, 1]
    """
    return np.corrcoef(roi_signals_filt.T)


def correlacion_parcial(roi_signals_filt):
    """
    Calcula la matriz de correlación parcial entre ROIs.

    Mide la correlación entre dos ROIs controlando el efecto lineal
    de todas las demás. Es un punto intermedio entre correlación simple
    y Graphical Lasso: reduce conexiones indirectas pero sin la penalización
    de sparsidad del GL.

    Parámetros
    ----------
    roi_signals_filt : array (T, n_rois) — señales filtradas

    Retorna
    -------
    pc_matrix : array (n_rois, n_rois)
    """
    measure   = ConnectivityMeasure(kind='partial correlation')
    pc_matrix = measure.fit_transform([roi_signals_filt])[0]
    return pc_matrix


def graphical_lasso(roi_signals_filt):
    """
    Calcula la matriz de precisión (covarianza inversa) usando Nilearn.

    Mide conectividad condicional: la entrada (i, j) es distinta de cero
    solo si la ROI i y la ROI j están conectadas directamente,
    descartando el efecto de todas las demás ROIs.

    Ventaja sobre correlación: separa conexiones directas de indirectas.
    Limitación: más sensible al número de timepoints vs. número de ROIs.

    Parámetros
    ----------
    roi_signals_filt : array (T, n_rois) — señales filtradas

    Retorna
    -------
    gl_matrix : array (n_rois, n_rois) — matriz de precisión

    Notas
    -----
    kind='precision' es el nombre actual en Nilearn para la covarianza
    inversa. En versiones antiguas se llamaba 'sparse inverse covariance'.
    """
    measure   = ConnectivityMeasure(kind='precision')
    gl_matrix = measure.fit_transform([roi_signals_filt])[0]
    return gl_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def umbralizar(matrix, threshold, absoluto=True):
    """
    Pone en cero las entradas de la matriz que no superan el umbral.

    Útil para limpiar la matriz de Graphical Lasso, donde entradas pequeñas
    pueden ser ruido numérico más que conectividad real.

    Parámetros
    ----------
    matrix    : array (n_rois, n_rois)
    threshold : float — umbral mínimo de valor absoluto (o valor directo)
    absoluto  : bool  — si True, el umbral se aplica sobre |matrix| (default True)

    Retorna
    -------
    array (n_rois, n_rois) — matriz umbralizada (no modifica la original)
    """
    result = matrix.copy()
    if absoluto:
        result[np.abs(result) < threshold] = 0
    else:
        result[result < threshold] = 0

    n_total   = matrix.size - matrix.shape[0]          # excluye diagonal
    n_nonzero = np.sum(result != 0) - matrix.shape[0]
    print(f'Umbral={threshold}: {n_nonzero} de {n_total} entradas no nulas '
          f'({100 * n_nonzero / n_total:.1f}%)')
    return result


def comparar_matrices(corr_matrix, gl_matrix, selected_rois, roi_names=None):
    """
    Imprime un resumen estadístico comparativo de ambas matrices.

    Parámetros
    ----------
    corr_matrix   : array (n_rois, n_rois)
    gl_matrix     : array (n_rois, n_rois)
    selected_rois : list[int]
    roi_names     : list[str] — opcional
    """
    n     = len(selected_rois)
    upper = np.triu_indices(n, k=1)

    corr_vals = corr_matrix[upper]
    gl_vals   = gl_matrix[upper]

    print("=== Correlación de Pearson ===")
    print(f"  Media  : {corr_vals.mean():.4f}")
    print(f"  Std    : {corr_vals.std():.4f}")
    print(f"  Min/Max: {corr_vals.min():.4f} / {corr_vals.max():.4f}")

    print("\n=== Graphical Lasso (precisión) ===")
    print(f"  Media  : {gl_vals.mean():.4f}")
    print(f"  Std    : {gl_vals.std():.4f}")
    print(f"  Min/Max: {gl_vals.min():.4f} / {gl_vals.max():.4f}")
    n_nonzero = np.sum(gl_vals != 0)
    print(f"  No nulas (tri. sup): {n_nonzero} de {len(gl_vals)} "
          f"({100 * n_nonzero / len(gl_vals):.1f}%)")