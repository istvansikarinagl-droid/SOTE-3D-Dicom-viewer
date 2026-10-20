// ============================================================
// CT volume rendering shader
// - Teljes képernyős raycasting egy 3D textúrán
// - Három tengely menti metszet sík, valódi belső nézettel
// - MIP (Maximum Intensity Projection) mód
// ============================================================

struct U {
    rot: mat4x4<f32>,
    bx: vec4<f32>,
    win: vec4<f32>,
    clip_a: vec4<f32>,
    clip_b: vec4<f32>,
    clip_c: vec4<f32>,
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
    let tn = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
    let tf = min(min(tmax.x, tmax.y), tmax.z);
    if (tf <= tn) {
        return vec4<f32>(0.02, 0.02, 0.03, 1.0);
    }

    // MIP mód: a sugár mentén a maximum HU értéket keressük
    let mip = u.clip_c.y > 0.5;

    // Ray marching
    let steps = 384;
    let dt = (tf - tn) / f32(steps);
    var acc = vec3<f32>(0.0);
    var alpha = 0.0;
    var max_v = -1.0;
    var t = tn;
    for (var i = 0; i < steps; i = i + 1) {
        let p = ro + rd * t;

        // Metszet síkok: a rejtett oldalon lévő mintákat kihagyjuk
        if (u.clip_a.x > 0.5) {
            let px = (u.clip_a.y * 2.0 - 1.0) * hb.x;
            let flip = u.clip_a.z > 0.5;
            let hidden = select(p.x > px, p.x < px, flip);
            if (hidden) {
                t = t + dt;
                continue;
            }
        }
        if (u.clip_a.w > 0.5) {
            let py = (u.clip_b.x * 2.0 - 1.0) * hb.y;
            let flip = u.clip_b.y > 0.5;
            let hidden = select(p.y > py, p.y < py, flip);
            if (hidden) {
                t = t + dt;
                continue;
            }
        }
        if (u.clip_b.z > 0.5) {
            let pz = (u.clip_b.w * 2.0 - 1.0) * hb.z;
            let flip = u.clip_c.x > 0.5;
            let hidden = select(p.z > pz, p.z < pz, flip);
            if (hidden) {
                t = t + dt;
                continue;
            }
        }

        // Világkoordináta -> textúra koordináta
        let tc = vec3<f32>(
            p.x / (2.0 * hb.x) + 0.5,
            p.z / (2.0 * hb.z) + 0.5,
            p.y / (2.0 * hb.y) + 0.5);

        // HU érték kiolvasása
        let v = textureSampleLevel(vol, smp, tc, 0.0).r;

        if (mip) {
            // MIP: csak a maximumot jegyezzük meg
            max_v = max(max_v, v);
        } else {
            // Normál volume rendering: felhalmozás
            let s = smoothstep(u.win.x, u.win.y, v);
            let a = s * u.win.z * dt * 20.0;
            let col = mix(vec3<f32>(0.75, 0.45, 0.35), vec3<f32>(1.0, 0.97, 0.9), s);
            acc = acc + (1.0 - alpha) * a * col;
            alpha = alpha + (1.0 - alpha) * a;
            if (alpha > 0.97) { break; }
        }
        t = t + dt;
    }

    if (mip) {
        // MIP eredmény: a maximum HU-t ablakozzuk
        let s = smoothstep(u.win.x, u.win.y, max_v);
        return vec4<f32>(vec3<f32>(s), 1.0);
    }

    let bg = vec3<f32>(0.02, 0.02, 0.03);
    return vec4<f32>(acc + (1.0 - alpha) * bg, 1.0);
}
