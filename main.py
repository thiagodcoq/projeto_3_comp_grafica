import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import wgpu
from rendercanvas.glfw import RenderCanvas, loop

import geometry
from math3d import look_at, normal_matrix, perspective, scale, translate

HERE = Path(__file__).parent
FORMAT = "bgra8unorm"   # sem sRGB: a correção gamma é feita no shader
DEPTH_FORMAT = "depth24plus"
CLEAR = (0.05, 0.06, 0.09, 1.0)

# ---------------------------------------------------------------- dispositivo
canvas = RenderCanvas(title="Projeto 3 - Cena base (MSAA 4x)", size=(1000, 700))
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
          "cube": make_mesh(*geometry.cube()),
          "plane": make_mesh(*geometry.plane())}

# ------------------------------------------------------------------- cena
# (malha, modelo, cor rgb, shininess, xadrez, intensidade especular)
objects_def = [
    ("plane", translate(0, 0, 0) @ scale(16, 1, 16), (0.8, 0.8, 0.8), 16, 1, 0.15),
    ("sphere", translate(-2.6, 1.0, 0.0) @ scale(1.0, 1.0, 1.0), (0.85, 0.15, 0.15), 64, 0, 0.8),
    ("sphere", translate(0.0, 0.7, 1.8) @ scale(0.7, 0.7, 0.7), (0.15, 0.7, 0.2), 128, 0, 1.0),
    ("sphere", translate(2.4, 1.3, -0.5) @ scale(1.3, 1.3, 1.3), (0.2, 0.3, 0.9), 32, 0, 0.6),
    ("cube", translate(0.2, 0.6, -2.4) @ scale(1.2, 1.2, 1.2), (0.9, 0.75, 0.2), 48, 0, 0.5),
    ("cube", translate(-1.2, 0.4, 2.8) @ scale(0.8, 0.8, 0.8), (0.7, 0.3, 0.8), 96, 0, 0.9),
]
lights = [  # posição, cor*intensidade
    ((-4.0, 5.0, 3.0), (1.3, 1.25, 1.15)),
    ((5.0, 4.0, 4.0), (0.35, 0.6, 1.2)),
    ((0.0, 6.0, -5.0), (1.1, 0.4, 0.4)),
]

# ------------------------------------------------------------------ buffers
def f32(*a):
    return np.concatenate([np.asarray(x, np.float32).ravel() for x in a]).astype(np.float32)

globals_buf = device.create_buffer(size=176, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

bgl_globals = device.create_bind_group_layout(entries=[{
    "binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
    "buffer": {"type": wgpu.BufferBindingType.uniform}}])
bgl_object = device.create_bind_group_layout(entries=[{
    "binding": 0, "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
    "buffer": {"type": wgpu.BufferBindingType.uniform}}])
globals_group = device.create_bind_group(layout=bgl_globals, entries=[
    {"binding": 0, "resource": {"buffer": globals_buf, "offset": 0, "size": 176}}])

draw_list = []
for mesh, model, color, shin, checker, spec in objects_def:
    data = f32(model.T, normal_matrix(model).T, (*color, shin), (checker, spec, 0, 0))
    buf = device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.UNIFORM)
    grp = device.create_bind_group(layout=bgl_object, entries=[
        {"binding": 0, "resource": {"buffer": buf, "offset": 0, "size": data.nbytes}}])
    draw_list.append((meshes[mesh], grp))

# ---------------------------------------------------------------- pipelines
shader = device.create_shader_module(code=(HERE / "shader.wgsl").read_text())
pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bgl_globals, bgl_object])

def make_pipeline(samples):
    return device.create_render_pipeline(
        layout=pipeline_layout,
        vertex={"module": shader, "entry_point": "vs_main", "buffers": [{
            "array_stride": 24, "step_mode": "vertex", "attributes": [
                {"format": "float32x3", "offset": 0, "shader_location": 0},
                {"format": "float32x3", "offset": 12, "shader_location": 1}]}]},
        fragment={"module": shader, "entry_point": "fs_main",
                  "targets": [{"format": FORMAT}]},
        primitive={"topology": "triangle-list", "cull_mode": "back", "front_face": "ccw"},
        depth_stencil={"format": DEPTH_FORMAT, "depth_write_enabled": True, "depth_compare": "less"},
        multisample={"count": samples})

pipelines = {1: make_pipeline(1), 4: make_pipeline(4)}
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
cam = {"yaw": 0.6, "pitch": 0.45, "dist": 11.0, "drag": None}

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
        if ev["key"] in ("m", "M"):
            samples = 1 if samples == 4 else 4
            canvas.set_title(f"Projeto 3 - Cena base (MSAA {samples}x)")
            print(f"MSAA {samples}x")
        elif ev["key"] in ("s", "S"):
            want_shot = True
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
def draw():
    global want_shot
    tex = context.get_current_texture()
    w, h = tex.size[0], tex.size[1]
    ensure_targets(w, h)

    eye = np.array([cam["dist"] * np.cos(cam["pitch"]) * np.sin(cam["yaw"]),
                    cam["dist"] * np.sin(cam["pitch"]),
                    cam["dist"] * np.cos(cam["pitch"]) * np.cos(cam["yaw"])], np.float32)
    vp = perspective(np.radians(45), w / h, 0.1, 100.0) @ look_at(eye, (0, 0.8, 0))
    gdata = f32(vp.T, (*eye, 1), *[(*p, 1) if k == 0 else (*c, 1)
                                   for p, c in lights for k in (0, 1)])
    device.queue.write_buffer(globals_buf, 0, gdata)

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
                                  "depth_load_op": "clear", "depth_store_op": "discard"})
    rp.set_pipeline(pipelines[samples])
    rp.set_bind_group(0, globals_group)
    for (vb, ib, n), grp in draw_list:
        rp.set_bind_group(1, grp)
        rp.set_vertex_buffer(0, vb)
        rp.set_index_buffer(ib, "uint32")
        rp.draw_indexed(n)
    rp.end()
    device.queue.submit([enc.finish()])
    if want_shot:
        want_shot = False
        screenshot(tex, w, h)

canvas.request_draw(draw)

if __name__ == "__main__":
    loop.run()
