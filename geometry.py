import numpy as np

# Cada vértice: posição (3) + normal (3), float32. Índices uint32.


def sphere(n_lat=48, n_lon=96):
    verts = []
    for i in range(n_lat + 1):
        th = np.pi * i / n_lat
        for j in range(n_lon + 1):
            ph = 2 * np.pi * j / n_lon
            p = (np.sin(th) * np.cos(ph), np.cos(th), np.sin(th) * np.sin(ph))
            verts.append((*p, *p))
    idx = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * (n_lon + 1) + j
            b = a + n_lon + 1
            idx += [a, a + 1, b, a + 1, b + 1, b]
    return np.array(verts, np.float32), np.array(idx, np.uint32)


def cube():
    faces = [  # normal, u, v (com u x v = normal -> CCW visto de fora)
        ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
        ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
        ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
    ]
    verts, idx = [], []
    for n, u, v in faces:
        n, u, v = map(np.array, (n, u, v))
        base = len(verts)
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = 0.5 * (n + su * u + sv * v)
            verts.append((*p, *n))
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return np.array(verts, np.float32), np.array(idx, np.uint32)


def plane():
    """Quadrado unitário no plano XZ (y=0), normal +Y."""
    v = [(-.5, 0, -.5), (.5, 0, -.5), (.5, 0, .5), (-.5, 0, .5)]
    verts = np.array([(*p, 0, 1, 0) for p in v], np.float32)
    # CCW visto de cima (+Y)
    return verts, np.array([0, 2, 1, 0, 3, 2], np.uint32)
