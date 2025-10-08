import numpy as np

def get_angles(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    v1 = np.asarray(vector1, dtype=float)
    v2 = np.asarray(vector2, dtype=float)
    dot = np.einsum('ij,ij->i', v1, v2)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(denom > 0, dot / denom, 0.0)
    ratio = np.clip(ratio, -1.0, 1.0)
    return np.degrees(np.arccos(ratio))

