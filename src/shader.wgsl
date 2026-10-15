struct U {
    rot: mat4x4<f32>,
    bx: vec4<f32>,
    win: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
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
    o.uv = p[i];
    return o;
}

@fragment
fn fs(in: VO) -> @location(0) vec4<f32> {
    let hb = u.bx.xyz;
    let ro = (u.rot * vec4<f32>(0.0, 0.0, u.bx.w, 1.0)).xyz;
    let rd = normalize((u.rot * vec4<f32>(in.uv.x * u.win.w * 0.5, in.uv.y * 0.5, -1.0, 0.0)).xyz);

    let t1 = (-hb - ro) / rd;
    let t2 = (hb - ro) / rd;
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);
    let tn = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
    let tf = min(min(tmax.x, tmax.y), tmax.z);
    if (tf <= tn) {
        return vec4<f32>(0.02, 0.02, 0.03, 1.0);
    }

    let steps = 384;
    let dt = (tf - tn) / f32(steps);
    var acc = vec3<f32>(0.0);
    var alpha = 0.0;
    var t = tn;
    for (var i = 0; i < steps; i = i + 1) {
        let p = ro + rd * t;
        let tc = vec3<f32>(
            p.x / (2.0 * hb.x) + 0.5,
            p.z / (2.0 * hb.z) + 0.5,
            p.y / (2.0 * hb.y) + 0.5);
        let v = textureSampleLevel(vol, smp, tc, 0.0).r;
        let s = smoothstep(u.win.x, u.win.y, v);
        let a = s * u.win.z * dt * 20.0;
        let col = mix(vec3<f32>(0.75, 0.45, 0.35), vec3<f32>(1.0, 0.97, 0.9), s);
        acc = acc + (1.0 - alpha) * a * col;
        alpha = alpha + (1.0 - alpha) * a;
        if (alpha > 0.97) { break; }
        t = t + dt;
    }
    let bg = vec3<f32>(0.02, 0.02, 0.03);
    return vec4<f32>(acc + (1.0 - alpha) * bg, 1.0);
}
