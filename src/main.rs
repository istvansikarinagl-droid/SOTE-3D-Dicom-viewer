// ============================================================
// SOTE 3D Dicom viewer - Rust + wgpu + egui
// - DICOM CT-sorozat betöltése mappából
// - GPU volume rendering (wgpu)
// - 2D MPR nézet: 2x2 elrendezés, panelenkénti zoom és pan
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
    window::WindowBuilder,
};

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

    // HU -> normalizált float16
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

// Uniform buffer a 3D shadernek
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    rot: [[f32; 4]; 4],
    bx: [f32; 4],
    win: [f32; 4],
    clipA: [f32; 4],
    clipB: [f32; 4],
    clipC: [f32; 4],
}

// Uniform buffer a 2D MPR shadernek
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

const PRESETS: [(&str, f32, f32); 4] = [
    ("Csont", 250.0, 1200.0),
    ("Erek (kontraszt)", 100.0, 700.0),
    ("Szervek", -50.0, 250.0),
    ("Tüdő", -900.0, -400.0),
];

fn norm(hu: f32) -> f32 {
    (hu + 1024.0) / 4096.0
}

fn main() {
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
        WindowBuilder::new()
            .with_title("SOTE 3D Dicom viewer")
            .build(&event_loop)
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
        },
        None,
    ))
    .unwrap();

    let size = window.inner_size();
    let mut config = surface
        .get_default_config(&adapter, size.width.max(1), size.height.max(1))
        .unwrap();
    surface.configure(&device, &config);

    // 3D textúra
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

    let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: std::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 2D MPR uniform bufferek
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
        vertex: wgpu::VertexState { module: &shader, entry_point: "vs", buffers: &[] },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs",
            targets: &[Some(config.format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let shader2 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("shader2d.wgsl").into()),
    });
    let pipeline2 = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pl),
        vertex: wgpu::VertexState { module: &shader2, entry_point: "vs", buffers: &[] },
        fragment: Some(wgpu::FragmentState {
            module: &shader2,
            entry_point: "fs",
            targets: &[Some(config.format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    // egui GUI
    let egui_ctx = egui::Context::default();
    let mut egui_state = State::new(
        egui_ctx.clone(),
        egui::ViewportId::default(),
        &window,
        None,
        None,
    );
    let mut egui_renderer = Renderer::new(&device, config.format, None, 1);

    // Állapotváltozók
    let (mut yaw, mut pitch, mut dist) = (0.6f32, 0.3f32, 3.2f32);
    let mut preset_idx = 0usize;
    let mut win_lo = PRESETS[0].1;
    let mut win_hi = PRESETS[0].2;
    let mut density = 1.0f32;
    let mut contrast_mode = false;
    let mut mpr_mode = false;
    let mut clip_x = (false, 0.5f32);
    let mut clip_y = (false, 0.5f32);
    let mut clip_z = (false, 0.5f32);
    let mut flip_x = false;
    let mut flip_y = false;
    let mut flip_z = false;

    // MPR panelenkénti zoom és pan
    let mut mpr_zoom = [1.0f32; 3];
    let mut mpr_pan = [[0.0f32; 2]; 3];
    let mut mpr_drag_panel: Option<usize> = None;

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
                        config.width = s.width.max(1);
                        config.height = s.height.max(1);
                        surface.configure(&device, &config);
                    }
                    WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                        let pressed = state == ElementState::Pressed;
                        if mpr_mode && pressed {
                            // Melyik MPR panelen kattintottunk? (2x2 rács)
                            let pw = (config.width / 2) as f64;
                            let ph = (config.height / 2) as f64;
                            let (cx, cy) = last;
                            let panel = if cy < ph {
                                if cx < pw { Some(0) } else { Some(1) }
                            } else if cx < pw {
                                Some(2)
                            } else {
                                None // jobb alsó üres
                            };
                            mpr_drag_panel = if egui_ctx.wants_pointer_input() { None } else { panel };
                            dragging = false;
                        } else {
                            mpr_drag_panel = None;
                            dragging = pressed && !egui_ctx.wants_pointer_input();
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        if let Some(pi) = mpr_drag_panel {
                            // Pan: az egér mozgásával arányosan, zoommal osztva
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
                            // Az aktív MPR panel zoomolása az egér pozíciója alapján
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
                        let raw_input = egui_state.take_egui_input(&window);
                        let full_output = egui_ctx.run(raw_input, |ctx| {
                            egui::TopBottomPanel::top("menu").show(ctx, |ui| {
                                ui.menu_button("Nézet", |ui| {
                                    for (i, (name, lo, hi)) in PRESETS.iter().enumerate() {
                                        if ui.button(*name).clicked() {
                                            preset_idx = i;
                                            win_lo = *lo;
                                            win_hi = *hi;
                                            ui.close_menu();
                                        }
                                    }
                                    ui.separator();
                                    ui.checkbox(&mut contrast_mode, "Kontrasztanyagos CT");
                                    ui.checkbox(&mut mpr_mode, "2D MPR nézet");
                                });
                            });

                            egui::SidePanel::right("panel").show(ctx, |ui| {
                                ui.heading("Megjelenítés");
                                ui.separator();

                                ui.label("Szövet típus");
                                egui::ComboBox::from_label("")
                                    .selected_text(PRESETS[preset_idx].0)
                                    .show_ui(ui, |ui| {
                                        for (i, (name, lo, hi)) in PRESETS.iter().enumerate() {
                                            if ui.selectable_label(preset_idx == i, *name).clicked() {
                                                preset_idx = i;
                                                win_lo = *lo;
                                                win_hi = *hi;
                                            }
                                        }
                                    });

                                ui.separator();
                                ui.checkbox(&mut mpr_mode, "2D MPR nézet");

                                ui.separator();
                                ui.label("HU ablak");
                                ui.add(egui::Slider::new(&mut win_lo, -1024.0..=2000.0).text("Alsó HU"));
                                ui.add(egui::Slider::new(&mut win_hi, -1024.0..=3000.0).text("Felső HU"));
                                if win_lo > win_hi { win_lo = win_hi; }

                                if !mpr_mode {
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
                            });
                        });

                        egui_state.handle_platform_output(&window, full_output.platform_output);

                        let (lo, hi) = if contrast_mode {
                            (win_lo - 200.0, win_hi + 400.0)
                        } else {
                            (win_lo, win_hi)
                        };

                        let u = Uniforms {
                            rot: rot_matrix(yaw, pitch),
                            bx: [hb[0], hb[1], hb[2], dist],
                            win: [norm(lo), norm(hi), density, config.width as f32 / config.height as f32],
                            clipA: [clip_x.0 as u32 as f32, clip_x.1, flip_x as u32 as f32, clip_y.0 as u32 as f32],
                            clipB: [clip_y.1, flip_y as u32 as f32, clip_z.0 as u32 as f32, clip_z.1],
                            clipC: [flip_z as u32 as f32, 0.0, 0.0, 0.0],
                        };
                        queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&u));

                        // MPR uniformok: szelet pozíció + zoom/pan
                        let u2 = [
                            Uniforms2D { p: [0.0, clip_z.1, norm(lo), norm(hi)], v: [mpr_zoom[0], mpr_pan[0][0], mpr_pan[0][1], 0.0] },
                            Uniforms2D { p: [1.0, clip_x.1, norm(lo), norm(hi)], v: [mpr_zoom[1], mpr_pan[1][0], mpr_pan[1][1], 0.0] },
                            Uniforms2D { p: [2.0, clip_y.1, norm(lo), norm(hi)], v: [mpr_zoom[2], mpr_pan[2][0], mpr_pan[2][1], 0.0] },
                        ];
                        for i in 0..3 {
                            queue.write_buffer(&u2bufs[i], 0, bytemuck::bytes_of(&u2[i]));
                        }

                        let Ok(frame) = surface.get_current_texture() else {
                            surface.configure(&device, &config);
                            return;
                        };
                        let fv = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                        let mut enc = device.create_command_encoder(&Default::default());

                        if mpr_mode {
                            // 2x2 MPR: bal felül axiális, jobb felül szagittális,
                            // bal alul koronális, jobb alul üres
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
                            pixels_per_point: window.scale_factor() as f32,
                        };
                        let tris = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
                        for (id, image_delta) in &full_output.textures_delta.set {
                            egui_renderer.update_texture(&device, &queue, *id, &image_delta);
                        }
                        egui_renderer.update_buffers(&device, &queue, &mut enc, &tris, &screen);
                        {
                            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
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
