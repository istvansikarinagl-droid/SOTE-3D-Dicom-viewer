// ============================================================
// CT volume rendering shader
// - Teljes képernyős raycasting egy 3D textúrán
// - Három tengely menti metszet sík támogatott
// ============================================================

struct U {
    rot: mat4x4<f32>,
    bx: vec4<f32>,
    win: vec4<f32>,
    clipA: vec4<f32>,
    clipB: vec4<f32>,
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

    // Kamera pozíció és sugárirány
    let ro = (u.rot * vec4<f32>(0.0, 0.0, u.bx.w, 1.0)).xyz;
    let rd = normalize((u.rot * vec4<f32>(
        in.uv.x * u.win.w * 0.5, in.uv.y * 0.5, -1.0, 0.0)).xyz);

    // Sugár-doboz metszés
    let t1 = (-hb - ro) / rd;
    let t2 = (hb - ro) / rd;
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);
    var tn = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
    var tf = min(min(tmax.x, tmax.y), tmax.z);
    if (tf <= tn) {
        return vec4<f32>(0.02, 0.02, 0.03, 1.0);
    }

    // Metszet síkok: a sík mögötti részt levágjuk
    if (u.clipA.x > 0.5) {
        let px = (u.clipA.y * 2.0 - 1.0) * hb.x;
        if (ro.x + rd.x * tn > px && rd.x > 0.0) { tn = max(tn, (px - ro.x) / rd.x); }
        if (ro.x + rd.x * tf < px && rd.x < 0.0) { tf = min(tf, (px - ro.x) / rd.x); }
    }
    if (u.clipA.z > 0.5) {
        let py = (u.clipA.w * 2.0 - 1.0) * hb.y;
        if (ro.y + rd.y * tn > py && rd.y > 0.0) { tn = max(tn, (py - ro.y) / rd.y); }
        if (ro.y + rd.y * tf < py && rd.y < 0.0) { tf = min(tf, (py - ro.y) / rd.y); }
    }
    if (u.clipB.x > 0.5) {
        let pz = (u.clipB.y * 2.0 - 1.0) * hb.z;
        if (ro.z + rd.z * tn > pz && rd.z > 0.0) { tn = max(tn, (pz - ro.z) / rd.z); }
        if (ro.z + rd.z * tf < pz && rd.z < 0.0) { tf = min(tf, (pz - ro.z) / rd.z); }
    }
    if (tf <= tn) {
        return vec4<f32>(0.02, 0.02, 0.03, 1.0);
    }

    // Ray marching
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
