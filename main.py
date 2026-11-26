import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import wgpu
from rendercanvas.glfw import RenderCanvas, loop

import envmap
import geometry
from ball_pool import BallPool
from math3d import look_at, normal_matrix, perspective, scale, shadow_matrix, translate

HERE = Path(__file__).parent
FORMAT = "bgra8unorm"   # sem sRGB: a correção gamma é feita no shader
DEPTH_FORMAT = "depth24plus-stencil8"   # profundidade + stencil (reflexão e sombras planares)
CLEAR = (0.05, 0.06, 0.09, 1.0)
FLOOR_OPACITY = 0.7     # o chão deixa passar 30% da imagem refletida
SHADOW_PLANE = (0, 1, 0, -0.005)   # plano y = 0.005: sombra um pouco acima do chão (evita z-fighting)

# ---------------------------------------------------------------- dispositivo
# modo contínuo: redesenha sem parar (animação); vsync desligado para medir o fps
canvas = RenderCanvas(title="Projeto 3", size=(1000, 700),
                      update_mode="continuous", max_fps=1000, vsync=False)
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync()
context = canvas.get_wgpu_context()
context.configure(device=device, format=FORMAT,
                  usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)

# ------------------------------------------------------------------- malhas
def make_mesh(verts, idx):
    vb = device.create_buffer_with_data(data=verts, usage=wgpu.BufferUsage.VERTEX)
    ib = device.create_buffer_with_data(data=idx, usage=wgpu.BufferUsage.INDEX)
    return vb, ib, len(idx)

meshes = {"sphere": make_mesh(*geometry.sphere()),
          "ball": make_mesh(*geometry.sphere(24, 48)),   # esfera mais leve para as bolas
          "cube": make_mesh(*geometry.cube()),
          "plane": make_mesh(*geometry.plane())}

# ------------------------------------------------------------------- cena
# (malha, modelo, cor rgb, shininess, xadrez, intensidade especular, refletividade do ambiente)
objects_def = [
    ("plane", translate(0, 0, 0) @ scale(16, 1, 16), (0.8, 0.8, 0.8), 16, 1, 0.15, 0.0),
    ("sphere", translate(-2.6, 1.0, 0.0) @ scale(1.0, 1.0, 1.0), (0.85, 0.15, 0.15), 64, 0, 0.8, 0.0),
    ("sphere", translate(0.0, 0.7, 1.8) @ scale(0.7, 0.7, 0.7), (0.15, 0.7, 0.2), 128, 0, 1.0, 0.0),
    ("sphere", translate(2.4, 1.3, -0.5) @ scale(1.3, 1.3, 1.3), (0.2, 0.3, 0.9), 32, 0, 0.6, 0.85),  # espelhada
    ("cube", translate(0.2, 0.6, -2.4) @ scale(1.2, 1.2, 1.2), (0.9, 0.75, 0.2), 48, 0, 0.5, 0.0),
    ("cube", translate(-1.2, 0.4, 2.8) @ scale(0.8, 0.8, 0.8), (0.7, 0.3, 0.8), 96, 0, 0.9, 0.0),
]
lights = [  # posição, cor*intensidade
    ((-4.0, 5.0, 3.0), (1.3, 1.25, 1.15)),
    ((5.0, 4.0, 4.0), (0.35, 0.6, 1.2)),
    ((0.0, 6.0, -5.0), (1.1, 0.4, 0.4)),
]
# sombra mais escura para a luz mais forte
SHADOW_ALPHA = [min(0.6, 0.5 * float(np.mean(c))) for _, c in lights]

# piscina de bolas: 4 paredes finas em volta de uma área quadrada de lado 2*POOL_HALF
POOL_CENTER, POOL_HALF, WALL_H, WALL_T = (2.9, 2.5), 1.0, 0.6, 0.12   # centro (x, z), metade do lado, altura, espessura
off = POOL_HALF + WALL_T / 2        # distância do centro da piscina até o meio de cada parede
side = 2 * POOL_HALF + 2 * WALL_T   # paredes em x são mais compridas para fechar os cantos
for dx, dz, sx, sz in ((off, 0, WALL_T, side), (-off, 0, WALL_T, side),
                       (0, off, 2 * POOL_HALF, WALL_T), (0, -off, 2 * POOL_HALF, WALL_T)):
    wall = translate(POOL_CENTER[0] + dx, WALL_H / 2, POOL_CENTER[1] + dz) @ scale(sx, WALL_H, sz)
    objects_def.append(("cube", wall, (0.9, 0.9, 0.92), 32, 0, 0.4, 0.0))

pool = BallPool(n=40, radius=0.22, center=POOL_CENTER, half=POOL_HALF)
first_ball = len(objects_def)
for color in pool.colors:   # modelo das bolas é atualizado a cada quadro
    objects_def.append(("ball", np.eye(4, dtype=np.float32), color, 64, 0, 0.7, 0.12))

# ------------------------------------------------------------------ buffers
def f32(*a):
    return np.concatenate([np.asarray(x, np.float32).ravel() for x in a]).astype(np.float32)

GLOBALS_SIZE = 256
OBJ_STRIDE = 256   # cada objeto ocupa 256 bytes do buffer de objetos (alinhamento de uniform)

env_view = envmap.make_cubemap(device, lights)
env_sampler = device.create_sampler(mag_filter="linear", min_filter="linear")

bgl_globals = device.create_bind_group_layout(entries=[
    {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
     "buffer": {"type": wgpu.BufferBindingType.uniform}},
    {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT,
     "texture": {"sample_type": "float", "view_dimension": "cube"}},
    {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {"type": "filtering"}}])
bgl_object = device.create_bind_group_layout(entries=[{
    "binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
    "buffer": {"type": wgpu.BufferBindingType.uniform}}])

def make_globals():
    buf = device.create_buffer(size=GLOBALS_SIZE, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
    grp = device.create_bind_group(layout=bgl_globals, entries=[
        {"binding": 0, "resource": {"buffer": buf, "offset": 0, "size": GLOBALS_SIZE}},
        {"binding": 1, "resource": env_view},
        {"binding": 2, "resource": env_sampler}])
    return buf, grp

main_globals = make_globals()                     # câmera normal
mirror_globals = make_globals()                   # cena espelhada no chão (reflexão planar)
shadow_globals = [make_globals() for _ in lights]  # uma projeção de sombra por luz

# todos os objetos num único buffer, reescrito a cada quadro (as bolas se movem)
obj_data = np.zeros((len(objects_def), OBJ_STRIDE // 4), np.float32)
objects_buf = device.create_buffer(size=obj_data.nbytes,
                                   usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

def set_model(i, model):
    obj_data[i, :32] = f32(model.T, normal_matrix(model).T)

draw_list = []
for i, (mesh, model, color, shin, checker, spec, refl) in enumerate(objects_def):
    set_model(i, model)
    obj_data[i, 32:40] = (*color, shin, checker, spec, refl, 1.0)
    grp = device.create_bind_group(layout=bgl_object, entries=[
        {"binding": 0, "resource": {"buffer": objects_buf, "offset": i * OBJ_STRIDE, "size": 160}}])
    draw_list.append((meshes[mesh], grp))
floor, others = draw_list[:1], draw_list[1:]

# ---------------------------------------------------------------- pipelines
shader = device.create_shader_module(code=(HERE / "shader.wgsl").read_text(encoding="utf-8"))
pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bgl_globals, bgl_object])
sky_layout = device.create_pipeline_layout(bind_group_layouts=[bgl_globals])
VERTEX_LAYOUT = {"array_stride": 24, "step_mode": "vertex", "attributes": [
    {"format": "float32x3", "offset": 0, "shader_location": 0},
    {"format": "float32x3", "offset": 12, "shader_location": 1}]}

def make_pipeline(samples, vs="vs_main", fs="fs_main", cull="back", blend=False, write_color=True,
                  depth_write=True, depth_compare="less", stencil="always", stencil_op="keep"):
    # stencil: passa se (referência <stencil> valor no buffer); se passar, aplica stencil_op
    face = {"compare": stencil, "fail_op": "keep", "depth_fail_op": "keep", "pass_op": stencil_op}
    target = {"format": FORMAT, "write_mask": wgpu.ColorWrite.ALL if write_color else 0}
    if blend:  # cor final = a * nova + (1 - a) * antiga; o alfa da tela fica 1
        target["blend"] = {
            "color": {"operation": "add", "src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha"},
            "alpha": {"operation": "add", "src_factor": "zero", "dst_factor": "one"}}
    sky = vs == "vs_sky"
    return device.create_render_pipeline(
        layout=sky_layout if sky else pipeline_layout,
        vertex={"module": shader, "entry_point": vs, "buffers": [] if sky else [VERTEX_LAYOUT]},
        fragment={"module": shader, "entry_point": fs, "targets": [target]},
        primitive={"topology": "triangle-list", "cull_mode": cull, "front_face": "ccw"},
        depth_stencil={"format": DEPTH_FORMAT, "depth_write_enabled": depth_write,
                       "depth_compare": depth_compare, "stencil_front": face, "stencil_back": face},
        multisample={"count": samples})

def make_pipelines(samples):
    return {
        # fundo: céu do mapa de ambiente, sem profundidade
        "sky": make_pipeline(samples, "vs_sky", "fs_sky", cull="none",
                             depth_write=False, depth_compare="always"),
        # marca no stencil (com a referência = 1) os pixels do chão, sem desenhar cor
        "mark": make_pipeline(samples, write_color=False, depth_write=False,
                              depth_compare="always", stencil_op="replace"),
        # reflexão planar: só onde o stencil == 1 (chão)
        "mirror_sky": make_pipeline(samples, "vs_sky", "fs_sky", cull="none", depth_write=False,
                                    depth_compare="always", stencil="equal"),
        # o espelho inverte a ordem dos vértices na tela: descarta as faces da frente
        "mirror": make_pipeline(samples, cull="front", stencil="equal"),
        # chão semitransparente por cima do reflexo
        "floor": make_pipeline(samples, blend=True),
        # sombra: só onde stencil == 1; ao passar zera o stencil, então cada pixel escurece 1 vez
        "shadow": make_pipeline(samples, fs="fs_shadow", cull="none", blend=True, depth_write=False,
                                stencil="equal", stencil_op="zero"),
        "main": make_pipeline(samples),
    }

pipelines = {1: make_pipelines(1), 4: make_pipelines(4)}
samples = 4

# ------------------------------------------------- texturas dependentes do tamanho
targets = {"size": None, "samples": None}

def ensure_targets(w, h):
    if targets["size"] == (w, h) and targets["samples"] == samples:
        return
    usage = wgpu.TextureUsage.RENDER_ATTACHMENT
    targets["depth"] = device.create_texture(
        size=(w, h, 1), format=DEPTH_FORMAT, usage=usage, sample_count=samples).create_view()
    targets["color"] = (device.create_texture(
        size=(w, h, 1), format=FORMAT, usage=usage, sample_count=samples).create_view()
        if samples > 1 else None)
    targets["size"], targets["samples"] = (w, h), samples

# ------------------------------------------------------------------ câmera
cam = {"yaw": 0.6, "pitch": 0.5, "dist": 12.5, "drag": None}
show = {"env": True, "shadows": True, "reflection": True, "physics": True}
TOGGLES = {"1": ("env", "reflexão de ambiente"), "2": ("shadows", "sombras planares"),
           "3": ("reflection", "reflexão planar"), "p": ("physics", "física")}

@canvas.add_event_handler("pointer_down", "pointer_up", "pointer_move", "wheel", "key_down")
def on_event(ev):
    global samples, want_shot
    t = ev["event_type"]
    if t == "pointer_down":
        cam["drag"] = (ev["x"], ev["y"])
    elif t == "pointer_up":
        cam["drag"] = None
    elif t == "pointer_move" and cam["drag"]:
        x0, y0 = cam["drag"]
        cam["yaw"] -= (ev["x"] - x0) * 0.008
        cam["pitch"] = float(np.clip(cam["pitch"] + (ev["y"] - y0) * 0.008, 0.05, 1.5))
        cam["drag"] = (ev["x"], ev["y"])
    elif t == "wheel":
        cam["dist"] = float(np.clip(cam["dist"] * (1.0 + 0.001 * ev["dy"]), 3.0, 30.0))
    elif t == "key_down":
        key = ev["key"].lower()
        if key == "m":
            samples = 1 if samples == 4 else 4
            print(f"MSAA {samples}x")
        elif key == "s":
            want_shot = True
        elif key in TOGGLES:
            name, label = TOGGLES[key]
            show[name] = not show[name]
            print(f"{label}: {'ON' if show[name] else 'OFF'}")
        elif key == "b":
            pool.reset()
        elif key == " ":
            pool.shake()
    canvas.request_draw()

want_shot = False

# ------------------------------------------------------------------ PNG
def save_png(path, rgba):
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data))
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

def screenshot(tex, w, h):
    bpr = (w * 4 + 255) // 256 * 256
    buf = device.create_buffer(size=bpr * h, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
    enc = device.create_command_encoder()
    enc.copy_texture_to_buffer({"texture": tex}, {"buffer": buf, "bytes_per_row": bpr, "rows_per_image": h}, (w, h, 1))
    device.queue.submit([enc.finish()])
    buf.map_sync(wgpu.MapMode.READ)
    img = np.frombuffer(buf.read_mapped(), np.uint8).reshape(h, bpr)[:, : w * 4].reshape(h, w, 4).copy()
    buf.unmap()
    img = img[:, :, [2, 1, 0, 3]]  # BGRA -> RGBA
    name = HERE / f"screenshot_msaa{samples}x_{time.strftime('%H%M%S')}.png"
    save_png(name, img)
    print("Screenshot salvo em", name)

# ------------------------------------------------------------------- desenho
clock = {"last": time.perf_counter(), "t0": time.perf_counter(), "frames": 0, "physics_ms": 0.0}

def globals_data(view_proj, inv_view_proj, eye, shadow_alpha=0.0):
    return f32(view_proj.T, inv_view_proj.T, (*eye, 1), (shadow_alpha, float(show["env"]), 0, 0),
               *[(*p, 1, *c, 1) for p, c in lights])

def draw_objects(rp, pipeline, globals_grp, objs):
    rp.set_pipeline(pipeline)
    rp.set_bind_group(0, globals_grp)
    for (vb, ib, n), grp in objs:
        rp.set_bind_group(1, grp)
        rp.set_vertex_buffer(0, vb)
        rp.set_index_buffer(ib, "uint32")
        rp.draw_indexed(n)

def draw_sky(rp, pipeline, globals_grp):
    rp.set_pipeline(pipeline)
    rp.set_bind_group(0, globals_grp)
    rp.draw(3)

def update(now):
    """Avança a física e atualiza os dados dos objetos."""
    dt = min(now - clock["last"], 1 / 30)   # limita o passo se a janela travar
    clock["last"] = now
    if show["physics"]:
        pool.step(dt)
    clock["physics_ms"] = 1000 * (time.perf_counter() - now)
    r = pool.r
    for i, p in enumerate(pool.pos):
        set_model(first_ball + i, translate(*p) @ scale(r, r, r))
    obj_data[0, 39] = FLOOR_OPACITY if show["reflection"] else 1.0   # opacidade do chão
    device.queue.write_buffer(objects_buf, 0, obj_data)

def draw():
    global want_shot
    now = time.perf_counter()
    update(now)
    tex = context.get_current_texture()
    w, h = tex.size[0], tex.size[1]
    ensure_targets(w, h)

    eye = np.array([cam["dist"] * np.cos(cam["pitch"]) * np.sin(cam["yaw"]),
                    cam["dist"] * np.sin(cam["pitch"]),
                    cam["dist"] * np.cos(cam["pitch"]) * np.cos(cam["yaw"])], np.float32)
    vp = perspective(np.radians(45), w / h, 0.1, 100.0) @ look_at(eye, (0, 0.8, 0))
    inv_vp = np.linalg.inv(vp)
    device.queue.write_buffer(main_globals[0], 0, globals_data(vp, inv_vp, eye))
    # reflexão: os objetos são espelhados em y (M = escala(1, -1, 1)) antes da câmera;
    # a iluminação usa o olho espelhado, como se a câmera estivesse embaixo do chão
    mirror = scale(1, -1, 1)
    device.queue.write_buffer(mirror_globals[0], 0,
                              globals_data(vp @ mirror, mirror @ inv_vp, eye * (1, -1, 1)))
    # sombra: os objetos são achatados no chão pela matriz de projeção a partir de cada luz
    for (buf, _), (pos, _), alpha in zip(shadow_globals, lights, SHADOW_ALPHA):
        device.queue.write_buffer(buf, 0, globals_data(vp @ shadow_matrix(pos, SHADOW_PLANE), inv_vp, eye, alpha))

    enc = device.create_command_encoder()
    view = tex.create_view()
    color_att = {"clear_value": CLEAR, "load_op": "clear", "store_op": "store"}
    if samples > 1:
        color_att |= {"view": targets["color"], "resolve_target": view}
    else:
        color_att |= {"view": view}
    rp = enc.begin_render_pass(
        color_attachments=[color_att],
        depth_stencil_attachment={"view": targets["depth"], "depth_clear_value": 1.0,
                                  "depth_load_op": "clear", "depth_store_op": "discard",
                                  "stencil_clear_value": 0, "stencil_load_op": "clear",
                                  "stencil_store_op": "discard"})
    p = pipelines[samples]
    rp.set_stencil_reference(1)
    g_main, g_mirror = main_globals[1], mirror_globals[1]

    # 1) fundo
    draw_sky(rp, p["sky"], g_main)
    # 2) reflexão planar: marca o chão no stencil e desenha a cena espelhada só ali
    if show["reflection"]:
        draw_objects(rp, p["mark"], g_main, floor)
        draw_sky(rp, p["mirror_sky"], g_mirror)
        draw_objects(rp, p["mirror"], g_mirror, others)
    # 3) chão (misturado com o reflexo)
    draw_objects(rp, p["floor"], g_main, floor)
    # 4) sombras planares: para cada luz, remarca o chão (a luz anterior zerou o stencil)
    #    e desenha os objetos achatados em preto semitransparente
    if show["shadows"]:
        for _, g_shadow in shadow_globals:
            draw_objects(rp, p["mark"], g_main, floor)
            draw_objects(rp, p["shadow"], g_shadow, others)
    # 5) objetos
    draw_objects(rp, p["main"], g_main, others)
    rp.end()
    device.queue.submit([enc.finish()])
    if want_shot:
        want_shot = False
        screenshot(tex, w, h)

    # fps e tempo da física no título da janela
    clock["frames"] += 1
    if now - clock["t0"] > 0.5:
        fps = clock["frames"] / (now - clock["t0"])
        canvas.set_title(f"Projeto 3 - MSAA {samples}x - {fps:.0f} fps - física {clock['physics_ms']:.2f} ms")
        clock["t0"], clock["frames"] = now, 0

canvas.request_draw(draw)

if __name__ == "__main__":
    print("Mouse: câmera | M: MSAA | 1: reflexão de ambiente | 2: sombras | 3: reflexão planar\n"
          "B: soltar as bolas | espaço: jogar as bolas para cima | P: pausar física | S: screenshot")
    loop.run()
