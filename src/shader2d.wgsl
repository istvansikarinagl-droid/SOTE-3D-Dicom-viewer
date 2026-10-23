// ============================================================
// 2D MPR shader
// - A 3D volumen egy síkját jeleníti meg 2D képként
// - Panelenkénti zoom és pan támogatás
// ============================================================

struct U2 {
    p: vec4<f32>, // x: sík típusa, y: szelet pozíció, z: ablak alja, w: ablak teteje
    v: vec4<f32>, // x: zoom, y: pan X, z: pan Y, w: tartalék
};
@group(0) @binding(0) var<uniform> u2: U2;
@group(0) @binding(1) var vol: texture_3d<f32>;
@group(0) @binding(2) var smp: sampler;

struct VO {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VO {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var o: VO;
    o.pos = vec4<f32>(p[i], 0.0, 1.0);
    o.uv = p[i] * 0.5 + 0.5;
    return o;
}

@fragment
fn fs(in: VO) -> @location(0) vec4<f32> {
    let zoom = max(u2.v.x, 0.05);
    let uv2 = (in.uv - vec2<f32>(0.5, 0.5)) / zoom + vec2<f32>(0.5, 0.5) + u2.v.yz;

    let plane = u2.p.x;
    let sl = u2.p.y;

    var tc: vec3<f32>;
    if (plane < 0.5) {
        tc = vec3<f32>(uv2.x, sl, uv2.y);
    } else if (plane < 1.5) {
        tc = vec3<f32>(sl, uv2.x, uv2.y);
    } else {
        tc = vec3<f32>(uv2.x, uv2.y, sl);
    }

    if (tc.x < 0.0 || tc.x > 1.0 || tc.y < 0.0 || tc.y > 1.0 || tc.z < 0.0 || tc.z > 1.0) {
        return vec4<f32>(vec3<f32>(0.0), 1.0);
    }

    let v = textureSampleLevel(vol, smp, tc, 0.0).r;

    let s = clamp((v - u2.p.z) / max(u2.p.w - u2.p.z, 0.0001), 0.0, 1.0);
    return vec4<f32>(vec3<f32>(s), 1.0);
}
