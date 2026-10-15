use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use dicom::dictionary_std::tags;
use dicom::object::open_file;
use dicom::pixeldata::PixelDecoder;
use wgpu::util::DeviceExt;
use winit::{
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

struct Slice {
    z: f64,
    rows: u32,
    cols: u32,
    sx: f64,
    sy: f64,
    hu: Vec<f32>,
}

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
        eprintln!(
            "{}: nem 16 bites ({} bájt, ts: {ts})",
            path.display(),
            raw.len()
        );
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

const PRESETS: [(&str, f32, f32); 3] = [
    ("Csont", 250.0, 1200.0),
    ("Lágyszövet", -100.0, 300.0),
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
            rfd::MessageDialog::new()
                .set_title("Hiba")
                .set_description(&e)
                .show();
            return;
        }
    };

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("CT 3D néző - egér: forgatás, görgő: zoom, 1/2/3: előbeállítás, +/-: sűrűség")
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

    let (mut yaw, mut pitch, mut dist) = (0.6f32, 0.3f32, 3.2f32);
    let mut preset = 0usize;
    let mut density = 1.0f32;
    let mut dragging = false;
    let mut last = (0.0f64, 0.0f64);

    event_loop
        .run(move |event, elwt| match event {
            Event::AboutToWait => window.request_redraw(),
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(s) => {
                    config.width = s.width.max(1);
                    config.height = s.height.max(1);
                    surface.configure(&device, &config);
                }
                WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                    dragging = state == ElementState::Pressed;
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if dragging {
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
                    dist = (dist * (1.0 - y * 0.08)).clamp(1.2, 10.0);
                }
                WindowEvent::KeyboardInput { event: k, .. } if k.state == ElementState::Pressed => {
                    if let PhysicalKey::Code(c) = k.physical_key {
                        match c {
                            KeyCode::Digit1 => preset = 0,
                            KeyCode::Digit2 => preset = 1,
                            KeyCode::Digit3 => preset = 2,
                            KeyCode::Equal | KeyCode::NumpadAdd => density = (density * 1.2).min(20.0),
                            KeyCode::Minus | KeyCode::NumpadSubtract => density = (density / 1.2).max(0.05),
                            _ => {}
                        }
                    }
                }
                WindowEvent::RedrawRequested => {
                    let (_, lo, hi) = (PRESETS[preset].0, PRESETS[preset].1, PRESETS[preset].2);
                    let u = Uniforms {
                        rot: rot_matrix(yaw, pitch),
                        bx: [hb[0], hb[1], hb[2], dist],
                        win: [norm(lo), norm(hi), density, config.width as f32 / config.height as f32],
                    };
                    queue.write_buffer(&ubuf, 0, bytemuck::bytes_of(&u));
                    let Ok(frame) = surface.get_current_texture() else {
                        surface.configure(&device, &config);
                        return;
                    };
                    let fv = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                    let mut enc = device.create_command_encoder(&Default::default());
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
                    queue.submit(Some(enc.finish()));
                    frame.present();
                }
                _ => {}
            },
            _ => {}
        })
        .unwrap();
}
