// ============================================================
// 3D Dicom viewer - Rust + wgpu + egui
// - DICOM CT-sorozat betöltése mappából
// - GPU volume rendering (wgpu)
// - 2D MPR nézet: 2x2 elrendezés, panelenkénti zoom és pan
// - MIP, Window/Level, izofelület, metszet síkok
// - Távolságmérés jobb egérgombbal, szelethez kötve
// - Kereszthajszál (crosshair), amely összeköti a három panelt
// - Képernyőkép mentés mentési dialógussal, mérésekkel együtt
// - Hibanaplózás (ct_viewer_error.log) és minimalizálás-védelem
// ============================================================

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use dicom::dictionary_std::tags;
use dicom::object::open_file;
use dicom::pixeldata::PixelDecoder;
use egui_wgpu::Renderer;
use egui_winit::State;
use wgpu::util::DeviceExt;
use winit::{
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

// Hibanapló: a program mappájába (aktuális könyvtár) ír, és a konzolra is
fn log_error(msg: &str) {
    use std::io::Write;
    let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("ct_viewer_error.log")
    {
        let _ = writeln!(f, "[{ts}] {msg}\n");
    }
    eprintln!("{msg}");
}

// Egy DICOM szelet adatai
struct Slice {
    z: f64,
    rows: u32,
    cols: u32,
    sx: f64,
    sy: f64,
    hu: Vec<f32>,
}

// A teljes 3D volumen
struct Volume {
    w: u32,
    h: u32,
    d: u32,
    spacing: [f64; 3],
    bytes: Vec<u8>,
}

// Egy kész távolságmérés (a mért szelethez kötve)
struct Measurement {
    panel: usize,
    pts: [[f32; 2]; 2], // textúra koordináták [0,1]
    mm: f32,
    color: egui::Color32,
    slice_idx: u32, // a panel szeletének voxel indexe a mérés pillanatában
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                collect_files(&p, out);
            } else {
                out.push(p);
            }
        }
    }
}

// Egy DICOM fájl beolvasása és HU értékek kinyerése
fn read_slice(path: &Path) -> Option<(String, Slice)> {
    let obj = match open_file(path) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{}: nem DICOM ({e})", path.display());
            return None;
        }
    };

    let ts = obj.meta().transfer_syntax().to_string();
    let series = obj
        .element(tags::SERIES_INSTANCE_UID)
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_string()))
        .unwrap_or_else(|| "ismeretlen".into());

    let pos = obj
        .element(tags::IMAGE_POSITION_PATIENT)
        .ok()
        .and_then(|e| e.to_multi_float64().ok());
    let Some(pos) = pos else {
        eprintln!("{}: nincs ImagePositionPatient (ts: {ts})", path.display());
        return None;
    };

    let rows = obj.element(tags::ROWS).ok()?.to_int::<u32>().ok()?;
    let cols = obj.element(tags::COLUMNS).ok()?.to_int::<u32>().ok()?;
    let signed = obj
        .element(tags::PIXEL_REPRESENTATION)
        .ok()
        .and_then(|e| e.to_int::<u32>().ok())
        .unwrap_or(0)
        == 1;
    let slope = obj
        .element(tags::RESCALE_SLOPE)
        .ok()
        .and_then(|e| e.to_float64().ok())
        .unwrap_or(1.0);
    let icpt = obj
        .element(tags::RESCALE_INTERCEPT)
        .ok()
        .and_then(|e| e.to_float64().ok())
        .unwrap_or(0.0);
    let ps = obj
        .element(tags::PIXEL_SPACING)
        .ok()
        .and_then(|e| e.to_multi_float64().ok())
        .unwrap_or_else(|| vec![1.0, 1.0]);

    let decoded = match obj.decode_pixel_data() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{}: dekódolási hiba (ts: {ts}): {e}", path.display());
            return None;
        }
    };
    let raw = decoded.data();
    if raw.len() != (rows * cols * 2) as usize {
        eprintln!("{}: nem 16 bites ({} bájt, ts: {ts})", path.display(), raw.len());
        return None;
    }
    let hu = raw
        .chunks_exact(2)
        .map(|c| {
            let v = if signed {
                i16::from_le_bytes([c[0], c[1]]) as f64
            } else {
                u16::from_le_bytes([c[0], c[1]]) as f64
            };
            (v * slope + icpt) as f32
        })
        .collect();
    Some((series, Slice { z: pos[2], rows, cols, sx: ps[1], sy: ps[0], hu }))
}

// A legtöbb szeletet tartalmazó sorozat betöltése
fn load_volume(dir: &Path) -> Result<Volume, String> {
    let mut files = Vec::new();
    collect_files(dir, &mut files);
    let mut groups: HashMap<String, Vec<Slice>> = HashMap::new();
    for f in &files {
        if let Some((s, sl)) = read_slice(f) {
            groups.entry(s).or_default().push(sl);
        }
    }
    let mut slices = groups
        .into_values()
        .max_by_key(|v| v.len())
        .ok_or("Nem találtam olvasható CT-szeleteket a mappában.")?;
    slices.sort_by(|a, b| a.z.partial_cmp(&b.z).unwrap());
    let (rows, cols) = (slices[0].rows, slices[0].cols);
    slices.retain(|s| s.rows == rows && s.cols == cols);
    let d = slices.len() as u32;
    let dz = if d > 1 { (slices[1].z - slices[0].z).abs() } else { 1.0 };

    let mut bytes = Vec::with_capacity((rows * cols * d * 2) as usize);
    for s in &slices {
        for &hu in &s.hu {
            let n = ((hu + 1024.0) / 4096.0).clamp(0.0, 1.0);
            bytes.extend_from_slice(&half::f16::from_f32(n).to_le_bytes());
        }
    }
    println!("{} szelet, {}x{}", d, cols, rows);
    Ok(Volume {
        w: cols,
        h: rows,
        d,
        spacing: [slices[0].sx, slices[0].sy, dz],
        bytes,
    })
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    rot: [[f32; 4]; 4],
    bx: [f32; 4],
    win: [f32; 4],
    clip_a: [f32; 4],
    clip_b: [f32; 4],
    clip_c: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms2D {
    p: [f32; 4],
    v: [f32; 4],
}

type M4 = [[f32; 4]; 4];

fn mul(a: &M4, b: &M4) -> M4 {
    let mut o = [[0.0; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            o[c][r] = (0..4).map(|k| a[k][r] * b[c][k]).sum();
        }
    }
    o
}

fn rot_matrix(yaw: f32, pitch: f32) -> M4 {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let ry = [
        [cy, 0.0, -sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [sy, 0.0, cy, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let rx = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cp, sp, 0.0],
        [0.0, -sp, cp, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    mul(&ry, &rx)
}

// Előbeállítások: (név, Window Width, Window Level)
const PRESETS: [(&str, f32, f32); 4] = [
    ("Csont", 2000.0, 500.0),
    ("Erek (kontraszt)", 600.0, 300.0),
    ("Szervek", 400.0, 50.0),
    ("Tüdő", 1500.0, -600.0),
];

// Mérőeszköz színek (egymás utáni mérésekhez)
const MEASURE_COLORS: [egui::Color32; 4] = [
    egui::Color32::YELLOW,
    egui::Color32::CYAN,
    egui::Color32::LIGHT_GREEN,
    egui::Color32::WHITE,
];

fn wl_to_lo_hi(width: f32, center: f32) -> (f32, f32) {
    (center - width / 2.0, center + width / 2.0)
}

fn norm(hu: f32) -> f32 {
    (hu + 1024.0) / 4096.0
}

// Panel rect: (x0, y0, w, h) pixelben
fn panel_rect(panel: usize, width: u32, height: u32) -> (f32, f32, f32, f32) {
    let pw = width as f32 / 2.0;
    let ph = height as f32 / 2.0;
    match panel {
        0 => (0.0, 0.0, pw, ph),
        1 => (pw, 0.0, width as f32 - pw, ph),
        _ => (0.0, ph, pw, height as f32 - ph),
    }
}

// Képernyő pont (pixel) -> textúra koordináta [0,1]
// A shader: tex_uv = (screen_uv - 0.5) / zoom + 0.5 + pan
fn screen_to_image(
    sx: f32,
    sy: f32,
    panel: usize,
    width: u32,
    height: u32,
    zoom: f32,
    pan: [f32; 2],
) -> [f32; 2] {
    let (x0, y0, w, h) = panel_rect(panel, width, height);
    let uvx = (sx - x0) / w;
    let uvy = 1.0 - (sy - y0) / h;
    [
        (uvx - 0.5) / zoom + 0.5 + pan[0],
        (uvy - 0.5) / zoom + 0.5 + pan[1],
    ]
}

// Textúra koordináta [0,1] -> képernyő pont (pixel)
fn image_to_screen(
    img: [f32; 2],
    panel: usize,
    width: u32,
    height: u32,
    zoom: f32,
    pan: [f32; 2],
) -> [f32; 2] {
    let (x0, y0, w, h) = panel_rect(panel, width, height);
    let uvx = (img[0] - 0.5 - pan[0]) * zoom + 0.5;
    let uvy = (img[1] - 0.5 - pan[1]) * zoom + 0.5;
    [x0 + uvx * w, y0 + (1.0 - uvy) * h]
}

// Voxel index és voxel-közép egy [0,1] tartománybeli csúszkaértékhez
fn slice_index(v: f32, n: u32) -> u32 {
    ((v.clamp(0.0, 1.0) * n as f32).floor() as u32).min(n.saturating_sub(1))
}

fn slice_center(idx: u32, n: u32) -> f32 {
    (idx as f32 + 0.5) / n as f32
}

// A csúszkák (clip_x, clip_y, clip_z) a textúra tengelyeit vezérlik:
//   clip_x -> tc.x (oszlop), clip_z -> tc.y (sor), clip_y -> tc.z (szelet)
// dims = [w, h, d]. Visszaad: (voxel indexek [x,y,z], voxel-közepek [x,y,z])
fn voxel_pos(clip_x: f32, clip_y: f32, clip_z: f32, dims: [u32; 3]) -> ([u32; 3], [f32; 3]) {
    let ix = slice_index(clip_x, dims[0]);
    let iy = slice_index(clip_z, dims[1]);
    let iz = slice_index(clip_y, dims[2]);
    (
        [ix, iy, iz],
        [
            slice_center(ix, dims[0]),
            slice_center(iy, dims[1]),
            slice_center(iz, dims[2]),
        ],
    )
}

// Kereszthajszál beállítása egy panelen kapott [0,1] koordinátából
// panel 0: (x, szelet), panel 1: (sor, szelet), panel 2: (x, sor)
fn set_crosshair(
    panel: usize,
    uv: [f32; 2],
    clip_x: &mut f32,
    clip_y: &mut f32,
    clip_z: &mut f32,
) {
    let u = uv[0].clamp(0.0, 1.0);
    let v = uv[1].clamp(0.0, 1.0);
    match panel {
        0 => {
            *clip_x = u;
            *clip_y = v;
        }
        1 => {
            *clip_z = u;
            *clip_y = v;
        }
        _ => {
            *clip_x = u;
            *clip_z = v;
        }
    }
}

// Az átfedő réteg (kereszthajszál + mérések) adatai
struct OverlayData<'a> {
    measurements: &'a [Measurement],
    pending: Option<(usize, [f32; 2], u32)>,
    pending_color: egui::Color32,
    cross: [f32; 3], // voxel-közepek [x,y,z] textúra koordinátában
    show_cross: bool,
    panel_idx: [u32; 3], // az egyes panelek aktuális szelet-indexe
    zoom: [f32; 3],
    pan: [[f32; 2]; 3],
    width: u32,
    height: u32,
    ppp: f32,
}

// Kirajzolja a kereszthajszálat és a méréseket (képernyőn és képernyőképen is)
fn draw_overlay(painter: &egui::Painter, o: &OverlayData) {
    let to_pt = |s: [f32; 2]| egui::Pos2::new(s[0] / o.ppp, s[1] / o.ppp);
    let panel_clip = |pi: usize| {
        let (x0, y0, w, h) = panel_rect(pi, o.width, o.height);
        egui::Rect::from_min_size(to_pt([x0, y0]), egui::vec2(w / o.ppp, h / o.ppp))
    };

    // Kereszthajszál
    if o.show_cross {
        let stroke = egui::Stroke::new(
            1.0_f32,
            egui::Color32::from_rgba_unmultiplied(255, 90, 90, 210),
        );
        for pi in 0..3 {
            let (x0, y0, w, h) = panel_rect(pi, o.width, o.height);
            let uv = match pi {
                0 => [o.cross[0], o.cross[2]],
                1 => [o.cross[1], o.cross[2]],
                _ => [o.cross[0], o.cross[1]],
            };
            let s = image_to_screen(uv, pi, o.width, o.height, o.zoom[pi], o.pan[pi]);
            let p = painter.with_clip_rect(panel_clip(pi));
            p.line_segment([to_pt([s[0], y0]), to_pt([s[0], y0 + h])], stroke);
            p.line_segment([to_pt([x0, s[1]]), to_pt([x0 + w, s[1]])], stroke);
        }
    }

    // Kész mérések (csak azon a szeleten, ahol készültek)
    for m in o.measurements {
        if m.slice_idx != o.panel_idx[m.panel] {
            continue;
        }
        let s0 = image_to_screen(m.pts[0], m.panel, o.width, o.height, o.zoom[m.panel], o.pan[m.panel]);
        let s1 = image_to_screen(m.pts[1], m.panel, o.width, o.height, o.zoom[m.panel], o.pan[m.panel]);
        let a = to_pt(s0);
        let b = to_pt(s1);
        let p = painter.with_clip_rect(panel_clip(m.panel));
        p.line_segment([a, b], egui::Stroke::new(2.0_f32, m.color));
        p.circle_filled(a, 4.0, m.color);
        p.circle_filled(b, 4.0, m.color);
        let mid = egui::Pos2::new((a.x + b.x) / 2.0, (a.y + b.y) / 2.0);
        let label = format!("{:.1} mm", m.mm);
        let font = egui::FontId::proportional(16.0);
        // Sötét árnyék a jobb olvashatóságért
        p.text(
            mid + egui::vec2(1.0, 1.0),
            egui::Align2::CENTER_BOTTOM,
            &label,
            font.clone(),
            egui::Color32::BLACK,
        );
        p.text(mid, egui::Align2::CENTER_BOTTOM, &label, font, m.color);
    }

    // Folyamatban lévő mérés első pontja
    if let Some((pi, p0, idx)) = o.pending {
        if idx == o.panel_idx[pi] {
            let s0 = image_to_screen(p0, pi, o.width, o.height, o.zoom[pi], o.pan[pi]);
            let p = painter.with_clip_rect(panel_clip(pi));
            p.circle_filled(to_pt(s0), 5.0, o.pending_color);
        }
    }
}

fn main() {
    // Panic esetén a hiba és a backtrace a naplófájlba is kerül
    std::panic::set_hook(Box::new(|info| {
        let bt = std::backtrace::Backtrace::force_capture();
        log_error(&format!("PANIC: {info}\n{bt}"));
    }));

    let dir = std::env::args().nth(1).map(PathBuf::from).or_else(|| {
        rfd::FileDialog::new()
            .set_title("Válaszd ki a CT DICOM mappáját")
            .pick_folder()
    });
    let Some(dir) = dir else { return };
    let vol = match load_volume(&dir) {
        Ok(v) => v,
        Err(e) => {
            rfd::MessageDialog::new().set_title("Hiba").set_description(&e).show();
            return;
        }
    };

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        event_loop
            .create_window(Window::default_attributes().with_title("3D Dicom viewer"))
            .unwrap(),
    );
    let instance = wgpu::Instance::default();
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .expect("Nincs megfelelő GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .unwrap();

    // A wgpu hibákat naplózzuk, ahelyett hogy a program összeomlana
    device.on_uncaptured_error(Box::new(|err| {
        log_error(&format!("wgpu hiba: {err}"));
    }));

    let size = window.inner_size();
    let mut config = surface
        .get_default_config(&adapter, size.width.max(2), size.height.max(2))
        .unwrap();
    surface.configure(&device, &config);

    let tex = device.create_texture_with_data(
        &queue,
        &wgpu::TextureDescriptor {
            label: Some("ct"),
            size: wgpu::Extent3d {
                width: vol.w,
                height: vol.h,
                depth_or_array_layers: vol.d,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &vol.bytes,
    );
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    // Fizikai kiterjedés mm-ben: [x (oszlop), szelet, sor]
    let ext = [
        vol.w as f64 * vol.spacing[0],
        vol.d as f64 * vol.spacing[2],
        vol.h as f64 * vol.spacing[1],
    ];
    let maxe = ext.iter().cloned().fold(0.0, f64::max);
    let hb = [
        (ext[0] / maxe) as f32,
        (ext[1] / maxe) as f32,
        (ext[2] / maxe) as f32,
    ];
    // Az egyes panelek két tengelyének fizikai kiterjedése (u, v) mm-ben
    let panel_ext: [[f32; 2]; 3] = [
        [ext[0] as f32, ext[1] as f32], // panel 0: x, szelet
        [ext[2] as f32, ext[1] as f32], // panel 1: sor, szelet
        [ext[0] as f32, ext[2] as f32], // panel 2: x, sor
    ];
    // Voxelszám a textúra tengelyei mentén: [x, y (sor), z (szelet)]
    let dims = [vol.w, vol.h, vol.d];

    let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: std::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let u2bufs: Vec<wgpu::Buffer> = (0..3)
        .map(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: std::mem::size_of::<Uniforms2D>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .collect();

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
        ],
    });

    let binds2: Vec<wgpu::BindGroup> = (0..3)
        .map(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: u2bufs[i].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                ],
            })
        })
        .collect();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(config.format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    let shader2 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("shader2d.wgsl").into()),
    });
    let pipeline2 = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &shader2,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader2,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(config.format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    // egui GUI
    let egui_ctx = egui::Context::default();
    let mut egui_state = State::new(
        egui_ctx.clone(),
        egui::ViewportId::default(),
        &window,
        None,
        None,
        None,
    );
    let mut egui_renderer = Renderer::new(&device, config.format, None, 1, false);

    // Külön egui környezet és renderer a képernyőkép átfedő rétegéhez
    // (mérések + kereszthajszál, a UI panelek nélkül)
    let overlay_ctx = egui::Context::default();
    let mut cap_renderer = Renderer::new(&device, config.format, None, 1, false);

    // Állapotváltozók
    let (mut yaw, mut pitch, mut dist) = (0.6f32, 0.3f32, 3.2f32);
    let mut preset_idx = 0usize;
    let mut win_width = PRESETS[0].1;
    let mut win_center = PRESETS[0].2;
    let mut density = 1.0f32;
    let mut contrast_mode = false;
    let mut mip_mode = false;
    let mut iso_mode = false;
    let mut iso_threshold = 0.31f32;
    let mut mpr_mode = false;
    let mut clip_x = (false, 0.5f32);
    let mut clip_y = (false, 0.5f32);
    let mut clip_z = (false, 0.5f32);
    let mut flip_x = false;
    let mut flip_y = false;
    let mut flip_z = false;

    let mut mpr_zoom = [1.0f32; 3];
    let mut mpr_pan = [[0.0f32; 2]; 3];
    let mut mpr_drag_panel: Option<usize> = None;

    // Kereszthajszál
    let mut show_cross = true;
    let mut cross_drag: Option<usize> = None;

    // Mérőeszköz állapot
    let mut measure_active = false;
    let mut measurements: Vec<Measurement> = Vec::new();
    // (panel, első pont, szelet-index)
    let mut measure_pending: Option<(usize, [f32; 2], u32)> = None;

    // Képernyőkép állapot
    let mut capture_requested = false;
    let mut pending_save: Option<Vec<u8>> = None;
    let mut save_status: Option<String> = None;

    // Minimalizált ablak: ilyenkor nem rajzolunk
    let mut minimized = false;

    let mut dragging = false;
    let mut last = (0.0f64, 0.0f64);

    event_loop
        .run(move |event, elwt| match event {
            Event::AboutToWait => window.request_redraw(),
            Event::WindowEvent { event, .. } => {
                let response = egui_state.on_window_event(&window, &event);
                if response.repaint {
                    window.request_redraw();
                }

                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::Resized(s) => {
                        if s.width == 0 || s.height == 0 {
                            // Minimalizált ablak: nincs mit rajzolni
                            minimized = true;
                        } else {
                            minimized = false;
                            config.width = s.width;
                            config.height = s.height;
                            surface.configure(&device, &config);
                        }
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state == ElementState::Pressed {
                            if let winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::F12) = event.physical_key {
                                capture_requested = true;
                            }
                        }
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        let pressed = state == ElementState::Pressed;
                        if !pressed {
                            mpr_drag_panel = None;
                            dragging = false;
                            cross_drag = None;
                            return;
                        }
                        if egui_ctx.wants_pointer_input() {
                            return;
                        }

                        let pw = (config.width / 2) as f64;
                        let ph = (config.height / 2) as f64;
                        let (cx, cy) = last;
                        let panel = if cy < ph {
                            if cx < pw { Some(0) } else { Some(1) }
                        } else if cx < pw {
                            Some(2)
                        } else {
                            None
                        };

                        match button {
                            MouseButton::Left => {
                                // Bal gomb: mozgatás (MPR) vagy forgatás (3D)
                                if mpr_mode {
                                    mpr_drag_panel = panel;
                                    dragging = false;
                                } else {
                                    dragging = true;
                                }
                            }
                            MouseButton::Right => {
                                if mpr_mode {
                                    if measure_active {
                                        // Jobb gomb: mérőpont lerakása
                                        if let Some(pi) = panel {
                                            let img = screen_to_image(
                                                cx as f32, cy as f32, pi,
                                                config.width, config.height,
                                                mpr_zoom[pi], mpr_pan[pi],
                                            );
                                            let (vi, _) = voxel_pos(clip_x.1, clip_y.1, clip_z.1, dims);
                                            let pidx = [vi[1], vi[0], vi[2]];
                                            match measure_pending {
                                                Some((sp, p0, sidx)) if sp == pi && sidx == pidx[pi] => {
                                                    // Fizikai távolság mm-ben a két tengely külön skálájával
                                                    let dx = (img[0] - p0[0]) * panel_ext[pi][0];
                                                    let dy = (img[1] - p0[1]) * panel_ext[pi][1];
                                                    let mm = (dx * dx + dy * dy).sqrt();
                                                    let color = MEASURE_COLORS[measurements.len() % 4];
                                                    measurements.push(Measurement {
                                                        panel: pi,
                                                        pts: [p0, img],
                                                        mm,
                                                        color,
                                                        slice_idx: pidx[pi],
                                                    });
                                                    measure_pending = None;
                                                }
                                                _ => {
                                                    // Új mérés kezdete (más panel vagy más szelet esetén is)
                                                    measure_pending = Some((pi, img, pidx[pi]));
                                                }
                                            }
                                        }
                                    } else if show_cross {
                                        // Mérés nélkül a jobb gomb a kereszthajszálat állítja
                                        if let Some(pi) = panel {
                                            let img = screen_to_image(
                                                cx as f32, cy as f32, pi,
                                                config.width, config.height,
                                                mpr_zoom[pi], mpr_pan[pi],
                                            );
                                            set_crosshair(pi, img, &mut clip_x.1, &mut clip_y.1, &mut clip_z.1);
                                            cross_drag = Some(pi);
                                        }
                                    }
                                }
                            }
                            MouseButton::Middle => {
                                // Középső gomb: kereszthajszál mindig
                                if mpr_mode && show_cross {
                                    if let Some(pi) = panel {
                                        let img = screen_to_image(
                                            cx as f32, cy as f32, pi,
                                            config.width, config.height,
                                            mpr_zoom[pi], mpr_pan[pi],
                                        );
                                        set_crosshair(pi, img, &mut clip_x.1, &mut clip_y.1, &mut clip_z.1);
                                        cross_drag = Some(pi);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        if let Some(pi) = cross_drag {
                            let img = screen_to_image(
                                position.x as f32, position.y as f32, pi,
                                config.width, config.height,
                                mpr_zoom[pi], mpr_pan[pi],
                            );
                            set_crosshair(pi, img, &mut clip_x.1, &mut clip_y.1, &mut clip_z.1);
                        } else if let Some(pi) = mpr_drag_panel {
                            let pw = (config.width / 2) as f32;
                            let ph = (config.height / 2) as f32;
                            let dx = (position.x - last.0) as f32 / (pw * mpr_zoom[pi]);
                            let dy = (position.y - last.1) as f32 / (ph * mpr_zoom[pi]);
                            mpr_pan[pi][0] -= dx;
                            mpr_pan[pi][1] += dy;
                        } else if dragging {
                            yaw -= (position.x - last.0) as f32 * 0.01;
                            pitch = (pitch + (position.y - last.1) as f32 * 0.01).clamp(-1.5, 1.5);
                        }
                        last = (position.x, position.y);
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        let y = match delta {
                            MouseScrollDelta::LineDelta(_, y) => y,
                            MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.02,
                        };
                        if mpr_mode {
                            let pw = (config.width / 2) as f64;
                            let ph = (config.height / 2) as f64;
                            let (cx, cy) = last;
                            let panel = if cy < ph {
                                if cx < pw { Some(0) } else { Some(1) }
                            } else if cx < pw {
                                Some(2)
                            } else {
                                None
                            };
                            if let Some(pi) = panel {
                                mpr_zoom[pi] = (mpr_zoom[pi] * (1.0 + y * 0.1)).clamp(1.0, 20.0);
                            }
                        } else {
                            dist = (dist * (1.0 - y * 0.08)).clamp(1.2, 10.0);
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        // Minimalizált vagy túl kicsi ablaknál nem rajzolunk
                        if minimized || config.width < 2 || config.height < 2 {
                            return;
                        }

                        let ppp = window.scale_factor() as f32;
                        let raw_input = egui_state.take_egui_input(&window);
                        let full_output = egui_ctx.run(raw_input, |ctx| {
                            // Képernyőkép mentés dialógussal
                            if let Some(pixels) = pending_save.take() {
                                let default_name = chrono::Local::now()
                                    .format("DICOM_screenshot_%Y%m%d_%H%M%S.png")
                                    .to_string();
                                if let Some(path) = rfd::FileDialog::new()
                                    .set_title("Képernyőkép mentése")
                                    .set_file_name(&default_name)
                                    .add_filter("PNG kép", &["png"])
                                    .save_file()
                                {
                                    let img = image::RgbaImage::from_raw(
                                        config.width, config.height, pixels,
                                    );
                                    match img {
                                        Some(img) => match img.save(&path) {
                                            Ok(_) => {
                                                save_status = Some(format!(
                                                    "Kép mentve: {}",
                                                    path.display()
                                                ));
                                            }
                                            Err(e) => {
                                                log_error(&format!("Mentési hiba: {e}"));
                                                save_status = Some(format!("Mentési hiba: {e}"));
                                            }
                                        },
                                        None => {
                                            save_status = Some("Nem sikerült a képet létrehozni".into());
                                        }
                                    }
                                } else {
                                    save_status = Some("Mentés megszakítva".into());
                                }
                            }

                            egui::TopBottomPanel::top("menu").show(ctx, |ui| {
                                ui.menu_button("Nézet", |ui| {
                                    for (i, (name, w, c)) in PRESETS.iter().enumerate() {
                                        if ui.button(*name).clicked() {
                                            preset_idx = i;
                                            win_width = *w;
                                            win_center = *c;
                                            ui.close_menu();
                                        }
                                    }
                                    ui.separator();
                                    ui.checkbox(&mut contrast_mode, "Kontrasztanyagos CT");
                                    ui.checkbox(&mut mip_mode, "MIP mód");
                                    ui.checkbox(&mut iso_mode, "Izofelület (csont felszín)");
                                    ui.checkbox(&mut mpr_mode, "2D MPR nézet");
                                    if mpr_mode {
                                        ui.checkbox(&mut measure_active, "Távolságmérés (jobb klikk)");
                                        ui.checkbox(&mut show_cross, "Kereszthajszál");
                                    }
                                });
                            });

                            egui::SidePanel::right("panel").show(ctx, |ui| {
                                ui.heading("Megjelenítés");
                                ui.separator();

                                ui.label("Szövet típus");
                                egui::ComboBox::from_label("")
                                    .selected_text(PRESETS[preset_idx].0)
                                    .show_ui(ui, |ui| {
                                        for (i, (name, w, c)) in PRESETS.iter().enumerate() {
                                            if ui.selectable_label(preset_idx == i, *name).clicked() {
                                                preset_idx = i;
                                                win_width = *w;
                                                win_center = *c;
                                            }
                                        }
                                    });

                                ui.separator();
                                ui.checkbox(&mut mpr_mode, "2D MPR nézet");
                                ui.checkbox(&mut mip_mode, "MIP mód");
                                ui.checkbox(&mut iso_mode, "Izofelület (csont felszín)");

                                if mpr_mode {
                                    ui.checkbox(&mut show_cross, "Kereszthajszál");
                                    ui.checkbox(&mut measure_active, "Távolságmérés (jobb klikk)");
                                    if measure_active {
                                        if ui.button("Összes mérés törlése").clicked() {
                                            measurements.clear();
                                            measure_pending = None;
                                        }
                                        ui.label(format!("Kész mérések: {}", measurements.len()));
                                        if measure_pending.is_some() {
                                            ui.label("Jobb klikk: második pont");
                                        } else {
                                            ui.label("Jobb klikk: első pont");
                                        }
                                    }
                                    if !measurements.is_empty() {
                                        let (vi_now, _) = voxel_pos(clip_x.1, clip_y.1, clip_z.1, dims);
                                        let pidx_now = [vi_now[1], vi_now[0], vi_now[2]];
                                        for (i, m) in measurements.iter().enumerate() {
                                            let here = m.slice_idx == pidx_now[m.panel];
                                            let text = format!(
                                                "{}. mérés: {:.1} mm (szelet {}){}",
                                                i + 1,
                                                m.mm,
                                                m.slice_idx + 1,
                                                if here { "" } else { " – másik szeleten" }
                                            );
                                            let color = if here { m.color } else { egui::Color32::GRAY };
                                            ui.colored_label(color, text);
                                        }
                                    }
                                }

                                if iso_mode {
                                    ui.separator();
                                    ui.label("Izofelület küszöb");
                                    ui.add(egui::Slider::new(&mut iso_threshold, 0.0..=1.0)
                                        .text("Küszöb"));
                                    ui.label(format!("Küszöb: {:.0} HU", iso_threshold * 4096.0 - 1024.0));
                                }

                                ui.separator();
                                ui.label("Window / Level");
                                ui.add(egui::Slider::new(&mut win_width, 1.0..=4000.0).text("Szélesség (W)"));
                                ui.add(egui::Slider::new(&mut win_center, -1000.0..=3000.0).text("Közép (L)"));
                                let (lo, hi) = wl_to_lo_hi(win_width, win_center);
                                ui.label(format!("HU tartomány: {:.0} … {:.0}", lo, hi));

                                if !mpr_mode && !mip_mode && !iso_mode {
                                    ui.separator();
                                    ui.label("Sűrűség (átlátszóság)");
                                    ui.add(egui::Slider::new(&mut density, 0.1..=10.0).logarithmic(true));
                                }

                                ui.separator();
                                ui.heading("Metszet síkok");
                                ui.checkbox(&mut clip_x.0, "X sík (bal-jobb)");
                                if clip_x.0 {
                                    ui.add(egui::Slider::new(&mut clip_x.1, 0.0..=1.0).text("X pozíció"));
                                    ui.checkbox(&mut flip_x, "Másik oldal (X)");
                                }
                                ui.checkbox(&mut clip_y.0, "Y sík (elöl-hátul)");
                                if clip_y.0 {
                                    ui.add(egui::Slider::new(&mut clip_y.1, 0.0..=1.0).text("Y pozíció"));
                                    ui.checkbox(&mut flip_y, "Másik oldal (Y)");
                                }
                                ui.checkbox(&mut clip_z.0, "Z sík (fej-láb)");
                                if clip_z.0 {
                                    ui.add(egui::Slider::new(&mut clip_z.1, 0.0..=1.0).text("Z pozíció"));
                                    ui.checkbox(&mut flip_z, "Másik oldal (Z)");
                                }

                                if mpr_mode {
                                    ui.separator();
                                    ui.heading("MPR panelek");
                                    for (i, name) in ["Axiális", "Szagittális", "Koronális"].iter().enumerate() {
                                        ui.label(*name);
                                        ui.add(egui::Slider::new(&mut mpr_zoom[i], 1.0..=20.0).text("Zoom"));
                                        if ui.button("Zoom visszaállítás").clicked() {
                                            mpr_zoom[i] = 1.0;
                                            mpr_pan[i] = [0.0, 0.0];
                                        }
                                    }
                                }

                                ui.separator();
                                if ui.button("Képernyőkép mentése (F12)").clicked() {
                                    capture_requested = true;
                                }

                                if let Some(status) = &save_status {
                                    ui.separator();
                                    ui.colored_label(egui::Color32::LIGHT_GREEN, status);
                                }
                            });

                            // Súgó panel a jobb alsó negyedben (2D MPR módban)
                            if mpr_mode {
                                egui::Area::new(egui::Id::new("help"))
                                    .anchor(egui::Align2::RIGHT_BOTTOM, [-12.0, -12.0])
                                    .show(ctx, |ui| {
                                        egui::Frame::group(ui.style())
                                            .fill(egui::Color32::from_black_alpha(180))
                                            .show(ui, |ui| {
                                                ui.set_max_width(300.0);
                                                ui.heading("Használat");
                                                ui.separator();
                                                ui.label("Bal egérgomb: kép mozgatása");
                                                ui.label("Görgő: zoom");
                                                ui.label("Középső gomb: kereszthajszál (húzható)");
                                                ui.label("Jobb gomb: mérőpont (mérés módban),");
                                                ui.label("  egyébként kereszthajszál");
                                                ui.separator();
                                                if measure_active {
                                                    ui.label("Távolságmérés: BE");
                                                    if measure_pending.is_some() {
                                                        ui.label("Következő: jobb klikk = 2. pont");
                                                    } else {
                                                        ui.label("Jobb klikk = 1. pont");
                                                    }
                                                } else {
                                                    ui.label("Távolságmérés: KI");
                                                }
                                                ui.label("A mérés csak a saját szeletén látszik.");
                                                ui.separator();
                                                ui.label("Mérőszínek sorrendben:");
                                                for (i, c) in MEASURE_COLORS.iter().enumerate() {
                                                    ui.colored_label(*c, format!("{}. mérés", i + 1));
                                                }
                                            });
                                    });
                            }

                            // Kereszthajszál és mérések kirajzolása
                            if mpr_mode {
                                let painter = ctx.layer_painter(egui::LayerId::new(
                                    egui::Order::Foreground,
                                    egui::Id::new("measure"),
                                ));
                                let (vi, ctr) = voxel_pos(clip_x.1, clip_y.1, clip_z.1, dims);
                                let data = OverlayData {
                                    measurements: &measurements,
                                    pending: measure_pending,
                                    pending_color: MEASURE_COLORS[measurements.len() % 4],
                                    cross: ctr,
                                    show_cross,
                                    panel_idx: [vi[1], vi[0], vi[2]],
                                    zoom: mpr_zoom,
                                    pan: mpr_pan,
                                    width: config.width,
                                    height: config.height,
                                    ppp,
                                };
                                draw_overlay(&painter, &data);
                            }
                        });

                        egui_state.handle_platform_output(&window, full_output.platform_output);

                        let (mut lo, mut hi) = wl_to_lo_hi(win_width, win_center);
                        if contrast_mode {
                            lo -= 200.0;
                            hi += 400.0;
                        }

                        let u = Uniforms {
                            rot: rot_matrix(yaw, pitch),
                            bx: [hb[0], hb[1], hb[2], dist],
                            win: [norm(lo), norm(hi), density, config.width as f32 / config.height as f32],
                            clip_a: [clip_x.0 as u32 as f32, clip_x.1, flip_x as u32 as f32, clip_y.0 as u32 as f32],
                            clip_b: [clip_y.1, flip_y as u32 as f32, clip_z.0 as u32 as f32, clip_z.1],
                            clip_c: [flip_z as u32 as f32, mip_mode as u32 as f32, iso_mode as u32 as f32, iso_threshold],
                        };
                        queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&u));

                        // Az MPR szeletek voxel-középre igazítva (nincs szeletek közti interpoláció)
                        let (vi, ctr) = voxel_pos(clip_x.1, clip_y.1, clip_z.1, dims);
                        let sl = [ctr[1], ctr[0], ctr[2]];
                        let u2 = [
                            Uniforms2D { p: [0.0, sl[0], norm(lo), norm(hi)], v: [mpr_zoom[0], mpr_pan[0][0], mpr_pan[0][1], 0.0] },
                            Uniforms2D { p: [1.0, sl[1], norm(lo), norm(hi)], v: [mpr_zoom[1], mpr_pan[1][0], mpr_pan[1][1], 0.0] },
                            Uniforms2D { p: [2.0, sl[2], norm(lo), norm(hi)], v: [mpr_zoom[2], mpr_pan[2][0], mpr_pan[2][1], 0.0] },
                        ];
                        for i in 0..3 {
                            queue.write_buffer(&u2bufs[i], 0, bytemuck::bytes_of(&u2[i]));
                        }

                        // Képernyőkép: külön encoderrel, a jelenet + mérések (UI panelek nélkül)
                        if capture_requested {
                            capture_requested = false;

                            let mut cap_enc = device.create_command_encoder(&Default::default());
                            let cap_tex = device.create_texture(&wgpu::TextureDescriptor {
                                label: Some("capture"),
                                size: wgpu::Extent3d {
                                    width: config.width,
                                    height: config.height,
                                    depth_or_array_layers: 1,
                                },
                                mip_level_count: 1,
                                sample_count: 1,
                                dimension: wgpu::TextureDimension::D2,
                                format: config.format,
                                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                                    | wgpu::TextureUsages::COPY_SRC,
                                view_formats: &[],
                            });
                            let cap_view = cap_tex.create_view(&wgpu::TextureViewDescriptor::default());

                            // 1. menet: a jelenet
                            {
                                let mut pass = cap_enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("capture"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &cap_view,
                                        resolve_target: None,
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                            store: wgpu::StoreOp::Store,
                                        },
                                    })],
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                });
                                if mpr_mode {
                                    let pw = config.width / 2;
                                    let ph = config.height / 2;
                                    let rects = [
                                        (0u32, 0u32, pw, ph),
                                        (pw, 0, config.width - pw, ph),
                                        (0, ph, pw, config.height - ph),
                                    ];
                                    for (i, (x, y, w, h)) in rects.iter().enumerate() {
                                        pass.set_viewport(*x as f32, *y as f32, *w as f32, *h as f32, 0.0, 1.0);
                                        pass.set_scissor_rect(*x, *y, *w, *h);
                                        pass.set_pipeline(&pipeline2);
                                        pass.set_bind_group(0, &binds2[i], &[]);
                                        pass.draw(0..3, 0..1);
                                    }
                                } else {
                                    pass.set_pipeline(&pipeline);
                                    pass.set_bind_group(0, &bind, &[]);
                                    pass.draw(0..3, 0..1);
                                }
                            }

                            // 2. menet: mérések és feliratok (csak 2D MPR módban)
                            if mpr_mode {
                                let cap_rect = egui::Rect::from_min_size(
                                    egui::Pos2::ZERO,
                                    egui::vec2(config.width as f32 / ppp, config.height as f32 / ppp),
                                );
                                let overlay_input = egui::RawInput {
                                    screen_rect: Some(cap_rect),
                                    ..Default::default()
                                };
                                overlay_ctx.set_pixels_per_point(ppp);
                                let data = OverlayData {
                                    measurements: &measurements,
                                    pending: None,
                                    pending_color: egui::Color32::WHITE,
                                    cross: ctr,
                                    // Ha a képernyőképen ne legyen kereszthajszál: false
                                    show_cross,
                                    panel_idx: [vi[1], vi[0], vi[2]],
                                    zoom: mpr_zoom,
                                    pan: mpr_pan,
                                    width: config.width,
                                    height: config.height,
                                    ppp,
                                };
                                let overlay_out = overlay_ctx.run(overlay_input, |ctx| {
                                    let painter = ctx.layer_painter(egui::LayerId::new(
                                        egui::Order::Foreground,
                                        egui::Id::new("cap_overlay"),
                                    ));
                                    draw_overlay(&painter, &data);
                                });
                                let cap_tris = overlay_ctx
                                    .tessellate(overlay_out.shapes, overlay_out.pixels_per_point);
                                for (id, delta) in &overlay_out.textures_delta.set {
                                    cap_renderer.update_texture(&device, &queue, *id, delta);
                                }
                                let cap_screen = egui_wgpu::ScreenDescriptor {
                                    size_in_pixels: [config.width, config.height],
                                    pixels_per_point: overlay_out.pixels_per_point,
                                };
                                cap_renderer.update_buffers(
                                    &device, &queue, &mut cap_enc, &cap_tris, &cap_screen,
                                );
                                {
                                    let pass = cap_enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                                        label: Some("capture_overlay"),
                                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                            view: &cap_view,
                                            resolve_target: None,
                                            ops: wgpu::Operations {
                                                load: wgpu::LoadOp::Load,
                                                store: wgpu::StoreOp::Store,
                                            },
                                        })],
                                        depth_stencil_attachment: None,
                                        timestamp_writes: None,
                                        occlusion_query_set: None,
                                    });
                                    let mut pass = pass.forget_lifetime();
                                    cap_renderer.render(&mut pass, &cap_tris, &cap_screen);
                                }
                                for id in &overlay_out.textures_delta.free {
                                    cap_renderer.free_texture(id);
                                }
                            }

                            let bytes_per_row = (config.width * 4).div_ceil(256) * 256;
                            let buf_size = (bytes_per_row * config.height) as u64;
                            let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("capture_out"),
                                size: buf_size,
                                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                                mapped_at_creation: false,
                            });
                            cap_enc.copy_texture_to_buffer(
                                cap_tex.as_image_copy(),
                                wgpu::TexelCopyBufferInfo {
                                    buffer: &out_buf,
                                    layout: wgpu::TexelCopyBufferLayout {
                                        offset: 0,
                                        bytes_per_row: Some(bytes_per_row),
                                        rows_per_image: Some(config.height),
                                    },
                                },
                                wgpu::Extent3d {
                                    width: config.width,
                                    height: config.height,
                                    depth_or_array_layers: 1,
                                },
                            );

                            queue.submit(Some(cap_enc.finish()));

                            let slice = out_buf.slice(..);
                            slice.map_async(wgpu::MapMode::Read, |_| {});
                            let _ = device.poll(wgpu::Maintain::Wait);
                            let data = slice.get_mapped_range();
                            let mut pixels = Vec::with_capacity((config.width * config.height * 4) as usize);
                            for row in 0..config.height {
                                let start = (row * bytes_per_row) as usize;
                                let end = start + (config.width * 4) as usize;
                                pixels.extend_from_slice(&data[start..end]);
                            }
                            drop(data);
                            out_buf.unmap();

                            // BGRA -> RGBA csere, ha a felület formátuma ezt kívánja, alfa = 255
                            let bgra = matches!(
                                config.format,
                                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
                            );
                            for px in pixels.chunks_exact_mut(4) {
                                if bgra {
                                    px.swap(0, 2);
                                }
                                px[3] = 255;
                            }
                            pending_save = Some(pixels);
                        }

                        // Normál render a képernyőre
                        let Ok(frame) = surface.get_current_texture() else {
                            surface.configure(&device, &config);
                            return;
                        };
                        let fv = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                        let mut enc = device.create_command_encoder(&Default::default());

                        if mpr_mode {
                            {
                                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("mpr"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &fv,
                                        resolve_target: None,
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                            store: wgpu::StoreOp::Store,
                                        },
                                    })],
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                });
                                let pw = config.width / 2;
                                let ph = config.height / 2;
                                let rects = [
                                    (0u32, 0u32, pw, ph),
                                    (pw, 0, config.width - pw, ph),
                                    (0, ph, pw, config.height - ph),
                                ];
                                for (i, (x, y, w, h)) in rects.iter().enumerate() {
                                    pass.set_viewport(*x as f32, *y as f32, *w as f32, *h as f32, 0.0, 1.0);
                                    pass.set_scissor_rect(*x, *y, *w, *h);
                                    pass.set_pipeline(&pipeline2);
                                    pass.set_bind_group(0, &binds2[i], &[]);
                                    pass.draw(0..3, 0..1);
                                }
                            }
                        } else {
                            {
                                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: None,
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &fv,
                                        resolve_target: None,
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                            store: wgpu::StoreOp::Store,
                                        },
                                    })],
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                });
                                pass.set_pipeline(&pipeline);
                                pass.set_bind_group(0, &bind, &[]);
                                pass.draw(0..3, 0..1);
                            }
                        }

                        // egui UI
                        let screen = egui_wgpu::ScreenDescriptor {
                            size_in_pixels: [config.width, config.height],
                            pixels_per_point: ppp,
                        };
                        let tris = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
                        for (id, image_delta) in &full_output.textures_delta.set {
                            egui_renderer.update_texture(&device, &queue, *id, &image_delta);
                        }
                        egui_renderer.update_buffers(&device, &queue, &mut enc, &tris, &screen);
                        {
                            let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("egui"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &fv,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });
                            let mut pass = pass.forget_lifetime();
                            egui_renderer.render(&mut pass, &tris, &screen);
                        }
                        for id in &full_output.textures_delta.free {
                            egui_renderer.free_texture(id);
                        }

                        queue.submit(Some(enc.finish()));
                        frame.present();
                    }
                    _ => {}
                }
            }
            _ => {}
        })
        .unwrap();
}
