import numpy as np

def normalize(vec):
    arr = np.array(vec, dtype=float)
    norm = np.linalg.norm(arr)
    return arr / (norm + 1e-12)

def cosine_sim(a, b):
    return float(np.dot(normalize(a), normalize(b)))
