"""Mapa de ambiente (cubemap) gerado por código: um estúdio escuro com painéis de luz."""
import numpy as np
import wgpu


def face_directions(face, n):
    """Direção de cada texel de uma face do cubemap (ordem +X, -X, +Y, -Y, +Z, -Z do WebGPU)."""
    t = (np.arange(n) + 0.5) / n * 2 - 1     # centro de cada texel em [-1, 1]
    s, t = np.meshgrid(t, t)                 # s cresce para a direita, t para baixo
    one = np.ones_like(s)
    d = [(one, -t, -s), (-one, -t, s),       # +X, -X
         (s, one, t), (s, -one, -t),         # +Y, -Y
         (s, -t, one), (-s, -t, -one)][face]  # +Z, -Z
    d = np.stack(d, axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def studio(d, lights):
    """Cor (linear) do ambiente vista na direção d."""
    y = d[..., 1:2]
    top, horizon = np.array((0.01, 0.012, 0.02)), np.array((0.06, 0.065, 0.085))
    sky = horizon + (top - horizon) * np.sqrt(np.clip(y, 0, 1))      # gradiente horizonte -> topo
    # abaixo do horizonte: chão cinza do estúdio, que se mistura com o horizonte ao longe
    ground = np.array((0.1, 0.1, 0.105))
    floor = ground + (horizon - ground) * np.exp(-6 * np.clip(-y, 0, 1))
    col = np.where(y > 0, sky, floor)
    # um painel circular na direção de cada luz da cena (como softboxes de estúdio)
    for pos, color in lights:
        c = np.asarray(pos) / np.linalg.norm(pos)
        k = smoothstep(np.cos(0.24), np.cos(0.20), d @ c)[..., None]
        col = col + k * 1.6 * np.asarray(color)
    return col


def make_cubemap(device, lights, n=256):
    """Cria a textura cúbica (6 camadas rgba16float) com o ambiente do estúdio."""
    data = np.zeros((6, n, n, 4), np.float16)
    for f in range(6):
        data[f, :, :, :3] = studio(face_directions(f, n), lights)
        data[f, :, :, 3] = 1.0
    tex = device.create_texture(size=(n, n, 6), format="rgba16float",
                                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
    device.queue.write_texture({"texture": tex}, data, {"bytes_per_row": n * 8, "rows_per_image": n}, (n, n, 6))
    return tex.create_view(dimension="cube")
