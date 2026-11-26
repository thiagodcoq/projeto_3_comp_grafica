import numpy as np


def perspective(fovy, aspect, near, far):
    """Projeção perspectiva com z de clip em [0, 1] (convenção WebGPU)."""
    f = 1.0 / np.tan(fovy / 2)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = far / (near - far)
    m[2, 3] = far * near / (near - far)
    m[3, 2] = -1
    return m


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = (np.asarray(v, dtype=np.float32) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def translate(x, y, z):
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = (x, y, z)
    return m


def scale(x, y, z):
    return np.diag([x, y, z, 1.0]).astype(np.float32)


def rotate_y(a):
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def normal_matrix(model):
    return np.linalg.inv(model).T.astype(np.float32)


def shadow_matrix(light, plane):
    """Projeta pontos sobre o plano (a, b, c, d) a partir de uma luz pontual: S = (p.l) I - l p^T."""
    l = np.array([*light, 1.0], dtype=np.float32)
    p = np.asarray(plane, dtype=np.float32)
    return (np.dot(p, l) * np.eye(4) - np.outer(l, p)).astype(np.float32)
