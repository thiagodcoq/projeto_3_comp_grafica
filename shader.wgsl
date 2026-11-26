struct Light {
    pos: vec4f,    // xyz = posição
    color: vec4f,  // rgb = cor * intensidade
};

struct Globals {
    view_proj: mat4x4f,
    inv_view_proj: mat4x4f,  // usada pelo céu para achar a direção de cada pixel
    eye: vec4f,
    params: vec4f,           // x = opacidade da sombra, y = 1 -> reflexão de ambiente ligada
    lights: array<Light, 3>,
};

struct Object {
    model: mat4x4f,
    normal_mat: mat4x4f,
    color: vec4f,   // rgb = cor difusa, a = shininess
    params: vec4f,  // x = 1 -> padrão xadrez, y = intensidade especular, z = refletividade, w = opacidade
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var env_tex: texture_cube<f32>;  // mapa de ambiente
@group(0) @binding(2) var env_samp: sampler;
@group(1) @binding(0) var<uniform> obj: Object;

struct VSOut {
    @builtin(position) clip: vec4f,
    @location(0) world_pos: vec3f,
    @location(1) normal: vec3f,
};

@vertex
fn vs_main(@location(0) pos: vec3f, @location(1) nrm: vec3f) -> VSOut {
    var o: VSOut;
    let wp = obj.model * vec4f(pos, 1.0);
    o.world_pos = wp.xyz;
    o.normal = (obj.normal_mat * vec4f(nrm, 0.0)).xyz;
    o.clip = g.view_proj * wp;
    return o;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
    var albedo = obj.color.rgb;
    if (obj.params.x > 0.5) {
        let c = floor(in.world_pos.x) + floor(in.world_pos.z);
        if (i32(c) % 2 == 0) { albedo = albedo * 0.35; }
    }
    let n = normalize(in.normal);
    let v = normalize(g.eye.xyz - in.world_pos);

    var col = albedo * 0.08; // ambiente
    for (var i = 0; i < 3; i++) {
        let L = g.lights[i];
        let d = L.pos.xyz - in.world_pos;
        let dist2 = dot(d, d);
        let l = d / sqrt(dist2);
        let att = 1.0 / (1.0 + 0.02 * dist2);
        let ndl = max(dot(n, l), 0.0);
        let h = normalize(l + v);
        let spec = select(0.0, pow(max(dot(n, h), 0.0), obj.color.a) * obj.params.y, ndl > 0.0);
        col += (albedo * ndl + vec3f(spec)) * L.color.rgb * att;
    }
    // reflexão de ambiente: reflete o vetor de visão na normal e busca a cor no cubemap
    let r = reflect(-v, n);
    let env = textureSample(env_tex, env_samp, r).rgb;
    col = mix(col, env, obj.params.z * g.params.y);
    // gamma (a swapchain é linear/unorm)
    return vec4f(pow(col, vec3f(1.0 / 2.2)), obj.params.w);
}

// ------------------------------------------------------------ céu (skybox)
struct SkyOut {
    @builtin(position) clip: vec4f,
    @location(0) ndc: vec2f,
};

@vertex
fn vs_sky(@builtin(vertex_index) i: u32) -> SkyOut {
    // um triângulo grande que cobre a tela inteira
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    var o: SkyOut;
    o.clip = vec4f(p, 1.0, 1.0);
    o.ndc = p;
    return o;
}

@fragment
fn fs_sky(in: SkyOut) -> @location(0) vec4f {
    // ponto no plano far em coordenadas do mundo -> direção a partir do olho
    let p = g.inv_view_proj * vec4f(in.ndc, 1.0, 1.0);
    let dir = p.xyz / p.w - g.eye.xyz;
    let col = textureSample(env_tex, env_samp, dir).rgb;
    return vec4f(pow(col, vec3f(1.0 / 2.2)), 1.0);
}

// ------------------------------------------------------------ sombra planar
@fragment
fn fs_shadow() -> @location(0) vec4f {
    // preto semitransparente: escurece o chão que já foi desenhado
    return vec4f(0.0, 0.0, 0.0, g.params.x);
}
