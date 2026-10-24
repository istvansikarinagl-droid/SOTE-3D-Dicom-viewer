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
fn load_window_icon() -> Option<winit::window::Icon> {
let img = image::load_from_memory(include_bytes!("../assets/icon.png"))
.ok()?
.into_rgba8();
let (w, h) = img.dimensions();
winit::window::Icon::from_rgba(img.into_raw(), w, h).ok()
}
#[derive(Clone, Copy, PartialEq)]
enum DisplayMode {
Auto,
Ct,
Mr,
}
struct Slice {
pos: [f64; 3],
iop: [f64; 6],
rows: u32,
cols: u32,
sx: f64,
sy: f64,
thickness: Option<f64>,
spacing_between: Option<f64>,
modality: String,
description: String,
px: Vec<f32>,
}
struct Volume {
w: u32,
h: u32,
d: u32,
spacing: [f64; 3], // [x, y (sor), z (szelet)] mm
data: Vec<u16>,    // f16 bitminták, normalizált [0,1]
nat_lo: f32,       // a 0 értéknek megfelelő natív érték
nat_hi: f32,       // az 1 értéknek megfelelő natív érték
is_ct: bool,
mr_presets: Vec<(String, f32, f32)>, // (név, szélesség, közép)
summary: String,
}
struct Measurement {
panel: usize,
pts: [[f32; 2]; 2],
mm: f32,
color: egui::Color32,
slice_idx: u32,
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
fn read_slice(path: &Path) -> Result<(String, Slice), (&'static str, String)> {
let obj = open_file(path).map_err(|e| ("nem DICOM", e.to_string()))?;
let ts = obj.meta().transfer_syntax().to_string();
let get_str = |tag| {
obj.element(tag)
.ok()
.and_then(|e| e.to_str().ok().map(|s| s.trim().to_string()))
};
let get_f64 = |tag| obj.element(tag).ok().and_then(|e| e.to_float64().ok());
let series = get_str(tags::SERIES_INSTANCE_UID).unwrap_or_else(|| "ismeretlen".into());
let modality = get_str(tags::MODALITY).unwrap_or_default();
let description = get_str(tags::SERIES_DESCRIPTION).unwrap_or_default();
let pos = obj
.element(tags::IMAGE_POSITION_PATIENT)
.ok()
.and_then(|e| e.to_multi_float64().ok())
.filter(|p| p.len() >= 3)
.ok_or(("hiányzó ImagePositionPatient", format!("ts: {ts}")))?;
let iop = match obj
.element(tags::IMAGE_ORIENTATION_PATIENT)
.ok()
.and_then(|e| e.to_multi_float64().ok())
.filter(|v| v.len() >= 6)
{
Some(v) => [v[0], v[1], v[2], v[3], v[4], v[5]],
None => [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
};
let rows = obj
.element(tags::ROWS)
.ok()
.and_then(|e| e.to_int::<u32>().ok())
.ok_or(("hiányzó Rows", String::new()))?;
let cols = obj
.element(tags::COLUMNS)
.ok()
.and_then(|e| e.to_int::<u32>().ok())
.ok_or(("hiányzó Columns", String::new()))?;
let signed = obj
.element(tags::PIXEL_REPRESENTATION)
.ok()
.and_then(|e| e.to_int::<u32>().ok())
.unwrap_or(0)
== 1;
let slope = get_f64(tags::RESCALE_SLOPE).unwrap_or(1.0);
let icpt = get_f64(tags::RESCALE_INTERCEPT).unwrap_or(0.0);
let ps = obj
.element(tags::PIXEL_SPACING)
.ok()
.and_then(|e| e.to_multi_float64().ok())
.filter(|v| v.len() >= 2)
.unwrap_or_else(|| vec![1.0, 1.0]);
let thickness = get_f64(tags::SLICE_THICKNESS);
let spacing_between = get_f64(tags::SPACING_BETWEEN_SLICES);
let decoded = obj
.decode_pixel_data()
.map_err(|e| ("dekódolási hiba", format!("{e} (ts: {ts})")))?;
let raw = decoded.data();
if raw.len() != (rows * cols * 2) as usize {
return Err(("nem 16 bites kép", format!("{} bájt, ts: {ts}", raw.len())));
}
let px: Vec<f32> = raw
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
Ok((
series,
Slice {
pos: [pos[0], pos[1], pos[2]],
iop,
rows,
cols,
sx: ps[1],
sy: ps[0],
thickness,
spacing_between,
modality,
description,
px,
},
))
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
[
a[1] * b[2] - a[2] * b[1],
a[2] * b[0] - a[0] * b[2],
a[0] * b[1] - a[1] * b[0],
]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn dominant(v: [f64; 3]) -> (usize, bool) {
let mut a = 0;
for i in 1..3 {
if v[i].abs() > v[a].abs() {
a = i;
}
}
(a, v[a] < 0.0)
}
fn make_hist(slices: &[Slice], lo: f32, hi: f32, nbins: usize) -> Vec<u64> {
let mut h = vec![0u64; nbins];
let scale = nbins as f32 / (hi - lo).max(1e-6);
for s in slices {
for &v in &s.px {
let i = (((v - lo) * scale) as isize).clamp(0, nbins as isize - 1) as usize;
h[i] += 1;
}
}
h
}
fn hist_quantile(h: &[u64], lo: f32, hi: f32, q: f64, start_bin: usize) -> f32 {
let total: u64 = h[start_bin.min(h.len())..].iter().sum();
if total == 0 {
return lo;
}
let target = ((q * total as f64) as u64).max(1);
let mut acc = 0u64;
for (i, &c) in h.iter().enumerate().skip(start_bin) {
acc += c;
if acc >= target {
return lo + (i as f32 + 0.5) / h.len() as f32 * (hi - lo);
}
}
hi
}
fn reorient_volume(
src: &[u16],
w: usize,
h: usize,
d: usize,
a: [usize; 3],
flip: [bool; 3],
) -> (Vec<u16>, [usize; 3]) {
let od = [w, h, d];
let ostride = [1usize, w, w * h];
let nd = [od[a[0]], od[a[1]], od[a[2]]];
let st = [ostride[a[0]], ostride[a[1]], ostride[a[2]]];
let mut out = vec![0u16; w * h * d];
let mut n = 0usize;
for k2 in 0..nd[2] {
let i2 = if flip[2] { nd[2] - 1 - k2 } else { k2 };
for k1 in 0..nd[1] {
let i1 = if flip[1] { nd[1] - 1 - k1 } else { k1 };
let base = i2 * st[2] + i1 * st[1];
for k0 in 0..nd[0] {
let i0 = if flip[0] { nd[0] - 1 - k0 } else { k0 };
out[n] = src[base + i0 * st[0]];
n += 1;
}
}
}
(out, nd)
}
fn load_volume(dir: &Path) -> Result<Volume, String> {
let mut report: Vec<String> = Vec::new();
report.push(format!("Betöltés ideje: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")));
report.push(format!("Mappa: {}", dir.display()));
let mut files = Vec::new();
collect_files(dir, &mut files);
report.push(format!("Talált fájlok: {}", files.len()));
let mut groups: HashMap<String, Vec<Slice>> = HashMap::new();
let mut skipped: HashMap<&'static str, usize> = HashMap::new();
let mut shown = 0usize;
for f in &files {
match read_slice(f) {
Ok((s, sl)) => groups.entry(s).or_default().push(sl),
Err((cat, detail)) => {
*skipped.entry(cat).or_insert(0) += 1;
if shown < 10 {
report.push(format!("  kihagyva: {} – {} {}", f.display(), cat, detail));
shown += 1;
}
}
}
}
for (cat, n) in &skipped {
report.push(format!("Kihagyott fájlok ({cat}): {n}"));
}
let mut overview: Vec<(usize, String)> = groups
.values()
.map(|v| {
let f = &v[0];
let desc = if f.description.is_empty() { "(nincs leírás)" } else { &f.description };
(
v.len(),
format!("  {} [{}] – {} kép, {}x{} px", desc, f.modality, v.len(), f.cols, f.rows),
)
})
.collect();
overview.sort_by(|a, b| b.0.cmp(&a.0));
report.push(format!("Talált sorozatok: {}", overview.len()));
for (_, line) in &overview {
report.push(line.clone());
}
let best_uid = groups
.iter()
.max_by_key(|(_, v)| v.len())
.map(|(k, _)| k.clone())
.ok_or("Nem találtam olvasható DICOM szeleteket a mappában.")?;
let mut slices = groups.remove(&best_uid).unwrap();
drop(groups);
report.push("Betöltött sorozat: a legtöbb képet tartalmazó".to_string());
let mut dim_count: HashMap<(u32, u32), usize> = HashMap::new();
for s in &slices {
*dim_count.entry((s.rows, s.cols)).or_insert(0) += 1;
}
let (rows, cols) = dim_count
.into_iter()
.max_by_key(|(_, c)| *c)
.map(|(k, _)| k)
.unwrap();
let before = slices.len();
slices.retain(|s| s.rows == rows && s.cols == cols);
if slices.len() != before {
report.push(format!("Eltérő méretű képek kihagyva: {}", before - slices.len()));
}
let iop = slices[0].iop;
let v0 = [iop[0], iop[1], iop[2]]; // az oszlopindex növekedésének iránya
let v1 = [iop[3], iop[4], iop[5]]; // a sorindex növekedésének iránya
let mut n = cross(v0, v1);
let nl = dot(n, n).sqrt();
if nl < 1e-6 {
report.push("FIGYELEM: érvénytelen ImageOrientationPatient, axiális irányt feltételezek".to_string());
n = [0.0, 0.0, 1.0];
} else {
n = [n[0] / nl, n[1] / nl, n[2] / nl];
}
slices.sort_by(|a, b| {
dot(a.pos, n)
.partial_cmp(&dot(b.pos, n))
.unwrap_or(std::cmp::Ordering::Equal)
});
let mut positions: Vec<f64> = Vec::with_capacity(slices.len());
let mut uniq: Vec<Slice> = Vec::with_capacity(slices.len());
let mut dups = 0usize;
for s in slices.drain(..) {
let p = dot(s.pos, n);
if let Some(&lp) = positions.last() {
if (p - lp).abs() < 1e-3 {
dups += 1;
continue;
}
}
positions.push(p);
uniq.push(s);
}
let slices = uniq;
if dups > 0 {
report.push(format!("Azonos pozíciójú (duplikált) képek kihagyva: {dups}"));
}
let d = slices.len();
if d < 2 {
report.push("FIGYELEM: kevesebb mint 2 szelet, a térfogat lapos lesz".to_string());
}
let diffs: Vec<f64> = positions.windows(2).map(|w| w[1] - w[0]).collect();
let median = if diffs.is_empty() {
None
} else {
let mut s = diffs.clone();
s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
Some(s[s.len() / 2])
};
let first = &slices[0];
let (dz, dz_src) = if let Some(m) = median.filter(|m| *m > 1e-6) {
(m, "ImagePositionPatient")
} else if let Some(sb) = first.spacing_between.map(f64::abs).filter(|v| *v > 1e-6) {
(sb, "SpacingBetweenSlices (tartalék)")
} else if let Some(t) = first.thickness.map(f64::abs).filter(|v| *v > 1e-6) {
(t, "SliceThickness (tartalék)")
} else {
(1.0, "alapértelmezés (1 mm)")
};
if !diffs.is_empty() {
let dmin = diffs.iter().cloned().fold(f64::MAX, f64::min);
let dmax = diffs.iter().cloned().fold(f64::MIN, f64::max);
if (dmax - dmin) > 0.05 * dz {
report.push(format!(
"FIGYELEM: egyenetlen szeletközök ({:.3} … {:.3} mm), a program egyenletest feltételez",
dmin, dmax
));
}
}
let dom = [dominant(v0), dominant(v1), dominant(n)];
let axis_names = ["x (jobb–bal)", "y (elöl–hátul)", "z (fej–láb)"];
let plane = match dom[2].0 {
0 => "szagittális",
1 => "koronális",
_ => "axiális",
};
for (i, name) in ["oszlopirány", "sorirány", "szeletirány"].iter().enumerate() {
let comp = [v0, v1, n][i];
let (ax, neg) = dom[i];
report.push(format!(
"  {} → {} {} (összetevő: {:.2})",
name,
axis_names[ax],
if neg { "negatív" } else { "pozitív" },
comp[ax].abs()
));
if comp[ax].abs() < 0.9 {
report.push(format!("  FIGYELEM: a(z) {name} ferde (<0,9), a megjelenítés közelítő"));
}
}
let axes = [dom[0].0, dom[1].0, dom[2].0];
let distinct = axes[0] != axes[1] && axes[0] != axes[2] && axes[1] != axes[2];
let (a, flip) = if distinct {
let a = [0usize, 1, 2].map(|k| axes.iter().position(|&x| x == k).unwrap());
let flip = [0usize, 1, 2].map(|k| dom[a[k]].1);
(a, flip)
} else {
report.push("FIGYELEM: a tengelyek nem egyértelműen feleltethetők meg, átrendezés nélkül jelenítem meg".to_string());
([0, 1, 2], [false; 3])
};
let identity = a == [0, 1, 2] && flip == [false; 3];
let w = cols as usize;
let h = rows as usize;
let (sx, sy) = (first.sx, first.sy);
let modality = first.modality.clone();
let description = first.description.clone();
let thickness = first.thickness;
let spacing_between = first.spacing_between;
let mut vmin = f32::MAX;
let mut vmax = f32::MIN;
for s in &slices {
for &v in &s.px {
if v < vmin {
vmin = v;
}
if v > vmax {
vmax = v;
}
}
}
if !(vmax > vmin) {
vmax = vmin + 1.0;
}
let h1 = make_hist(&slices, vmin, vmax, 16384);
let q_hi = hist_quantile(&h1, vmin, vmax, 0.9995, 0);
let hi_clip = if q_hi > vmin + 1.0 { q_hi } else { vmax };
let h2 = make_hist(&slices, vmin, hi_clip, 4096);
let bg_bin = (0.06 * 4096.0) as usize; // a sötét háttér kihagyása
let fp02 = hist_quantile(&h2, vmin, hi_clip, 0.02, bg_bin);
let fp50 = hist_quantile(&h2, vmin, hi_clip, 0.50, bg_bin);
let fp98 = hist_quantile(&h2, vmin, hi_clip, 0.98, bg_bin);
let fp995 = hist_quantile(&h2, vmin, hi_clip, 0.995, bg_bin);
let is_ct = modality.eq_ignore_ascii_case("CT") || (modality.is_empty() && vmin <= -500.0);
let (nat_lo, nat_hi) = if is_ct { (-1024.0f32, 3072.0f32) } else { (vmin, hi_clip) };
let nat_range = (nat_hi - nat_lo).max(1.0);
report.push(format!("Nyers értéktartomány: {:.1} … {:.1}", vmin, vmax));
report.push(format!(
"Előtér-percentilisek: P2 = {:.1}, P50 = {:.1}, P98 = {:.1}, P99,5 = {:.1}",
fp02, fp50, fp98, fp995
));
report.push(format!(
"Normalizálás: {} ({:.1} … {:.1})",
if is_ct { "CT, rögzített HU-skála" } else { "adatvezérelt (min … P99,95)" },
nat_lo,
nat_hi
));
let wl = |lo: f32, hi: f32| {
let hi = if hi > lo + 1.0 { hi } else { lo + 1.0 };
(hi - lo, (hi + lo) / 2.0)
};
let (w1, l1) = wl(nat_lo, fp995);
let (w2, l2) = wl(fp02, fp98);
let (w3, l3) = wl(fp50, fp995);
let (w4, l4) = wl(nat_lo, nat_hi);
let mr_presets = vec![
("Automatikus".to_string(), w1, l1),
("Lágyrész kontraszt (P2–P98)".to_string(), w2, l2),
("Világos szerkezetek (P50–P99,5)".to_string(), w3, l3),
("Teljes tartomány".to_string(), w4, l4),
];
let inv = 1.0 / nat_range;
let mut orig: Vec<u16> = Vec::with_capacity(w * h * d);
for s in slices {
orig.extend(s.px.iter().map(|&v| {
half::f16::from_f32(((v - nat_lo) * inv).clamp(0.0, 1.0)).to_bits()
}));
}
let (data, nd) = if identity {
report.push("Átrendezés: nem szükséges".to_string());
(orig, [w, h, d])
} else {
let (out, nd) = reorient_volume(&orig, w, h, d, a, flip);
report.push(format!(
"Átrendezés: igen (új tengelyek az eredetiből: {:?}, megfordítva: {:?})",
a, flip
));
(out, nd)
};
let ospacing = [sx, sy, dz];
let spacing = [ospacing[a[0]], ospacing[a[1]], ospacing[a[2]]];
report.push(format!("Modalitás: {}", if modality.is_empty() { "ismeretlen" } else { &modality }));
report.push(format!("Sorozat leírása: {}", if description.is_empty() { "(nincs)" } else { &description }));
report.push(format!("Szeletek száma: {d}, képméret: {cols}x{rows} px"));
report.push(format!("Szelet-sík: {plane}"));
report.push(format!("Pixelméret: {:.3} x {:.3} mm (oszlop x sor)", sx, sy));
report.push(format!("Szeletköz: {:.3} mm (forrás: {})", dz, dz_src));
if let Some(t) = thickness {
report.push(format!("SliceThickness: {:.3} mm", t));
}
if let Some(sb) = spacing_between {
report.push(format!("SpacingBetweenSlices: {:.3} mm", sb));
}
report.push(format!(
"Végső voxelméret: {}x{}x{} (x, y, z), {:.3} x {:.3} x {:.3} mm",
nd[0], nd[1], nd[2], spacing[0], spacing[1], spacing[2]
));
report.push(format!(
"Térfogat mérete: {:.0} x {:.0} x {:.0} mm",
nd[0] as f64 * spacing[0],
nd[1] as f64 * spacing[1],
nd[2] as f64 * spacing[2]
));
let info = report.join("\n");
println!("{info}");
let _ = std::fs::write("ct_viewer_load.log", &info);
let summary = format!(
"{} {}\n{}, {} szelet, {}x{} px\nSzeletköz: {:.2} mm\nPixelméret: {:.2} x {:.2} mm\nÁtrendezés: {}\nFelismert mód: {}",
if modality.is_empty() { "?" } else { &modality },
description,
plane,
d,
cols,
rows,
dz,
sx,
sy,
if identity { "nem" } else { "igen" },
if is_ct { "CT" } else { "MR / egyéb" }
);
Ok(Volume {
w: nd[0] as u32,
h: nd[1] as u32,
d: nd[2] as u32,
spacing,
data,
nat_lo,
nat_hi,
is_ct,
mr_presets,
summary,
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
const PRESETS: [(&str, f32, f32); 4] = [
("Csont", 2000.0, 500.0),
("Erek (kontraszt)", 600.0, 300.0),
("Szervek", 400.0, 50.0),
("Tüdő", 1500.0, -600.0),
];
fn build_presets(mode: DisplayMode, mr: &[(String, f32, f32)]) -> Vec<(String, f32, f32)> {
match mode {
DisplayMode::Ct => PRESETS.iter().map(|(n, w, l)| (n.to_string(), *w, *l)).collect(),
_ => mr.to_vec(),
}
}
fn default_iso(mode: DisplayMode) -> f32 {
match mode {
DisplayMode::Ct => 0.31,
_ => 0.12,
}
}
const MEASURE_COLORS: [egui::Color32; 4] = [
egui::Color32::YELLOW,
egui::Color32::CYAN,
egui::Color32::LIGHT_GREEN,
egui::Color32::WHITE,
];
fn wl_to_lo_hi(width: f32, center: f32) -> (f32, f32) {
(center - width / 2.0, center + width / 2.0)
}
fn panel_rect(panel: usize, width: u32, height: u32) -> (f32, f32, f32, f32) {
let pw = width as f32 / 2.0;
let ph = height as f32 / 2.0;
match panel {
0 => (0.0, 0.0, pw, ph),
1 => (pw, 0.0, width as f32 - pw, ph),
_ => (0.0, ph, pw, height as f32 - ph),
}
}
fn screen_to_image(sx: f32, sy: f32, panel: usize, width: u32, height: u32, zoom: f32, pan: [f32; 2]) -> [f32; 2] {
let (x0, y0, w, h) = panel_rect(panel, width, height);
let uvx = (sx - x0) / w;
let uvy = 1.0 - (sy - y0) / h;
[(uvx - 0.5) / zoom + 0.5 + pan[0], (uvy - 0.5) / zoom + 0.5 + pan[1]]
}
fn image_to_screen(img: [f32; 2], panel: usize, width: u32, height: u32, zoom: f32, pan: [f32; 2]) -> [f32; 2] {
let (x0, y0, w, h) = panel_rect(panel, width, height);
let uvx = (img[0] - 0.5 - pan[0]) * zoom + 0.5;
let uvy = (img[1] - 0.5 - pan[1]) * zoom + 0.5;
[x0 + uvx * w, y0 + (1.0 - uvy) * h]
}
fn slice_index(v: f32, n: u32) -> u32 {
((v.clamp(0.0, 1.0) * n as f32).floor() as u32).min(n.saturating_sub(1))
}
fn slice_center(idx: u32, n: u32) -> f32 {
(idx as f32 + 0.5) / n as f32
}
fn voxel_pos(clip_x: f32, clip_y: f32, clip_z: f32, dims: [u32; 3]) -> ([u32; 3], [f32; 3]) {
let ix = slice_index(clip_x, dims[0]);
let iy = slice_index(clip_z, dims[1]);
let iz = slice_index(clip_y, dims[2]);
(
[ix, iy, iz],
[slice_center(ix, dims[0]), slice_center(iy, dims[1]), slice_center(iz, dims[2])],
)
}
fn set_crosshair(panel: usize, uv: [f32; 2], clip_x: &mut f32, clip_y: &mut f32, clip_z: &mut f32) {
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
struct OverlayData<'a> {
measurements: &'a [Measurement],
pending: Option<(usize, [f32; 2], u32)>,
pending_color: egui::Color32,
cross: [f32; 3],
show_cross: bool,
panel_idx: [u32; 3],
zoom: [f32; 3],
pan: [[f32; 2]; 3],
width: u32,
height: u32,
ppp: f32,
}
fn draw_overlay(painter: &egui::Painter, o: &OverlayData) {
let to_pt = |s: [f32; 2]| egui::Pos2::new(s[0] / o.ppp, s[1] / o.ppp);
let panel_clip = |pi: usize| {
let (x0, y0, w, h) = panel_rect(pi, o.width, o.height);
egui::Rect::from_min_size(to_pt([x0, y0]), egui::vec2(w / o.ppp, h / o.ppp))
};
if o.show_cross {
let stroke = egui::Stroke::new(1.0_f32, egui::Color32::from_rgba_unmultiplied(255, 90, 90, 210));
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
p.text(mid + egui::vec2(1.0, 1.0), egui::Align2::CENTER_BOTTOM, &label, font.clone(), egui::Color32::BLACK);
p.text(mid, egui::Align2::CENTER_BOTTOM, &label, font, m.color);
}
if let Some((pi, p0, idx)) = o.pending {
if idx == o.panel_idx[pi] {
let s0 = image_to_screen(p0, pi, o.width, o.height, o.zoom[pi], o.pan[pi]);
let p = painter.with_clip_rect(panel_clip(pi));
p.circle_filled(to_pt(s0), 5.0, o.pending_color);
}
}
}
fn project_3d(p: [f32; 3], rot: &M4, dist: f32, width: u32, height: u32) -> Option<[f32; 2]> {
let l = [0usize, 1, 2].map(|c| rot[c][0] * p[0] + rot[c][1] * p[1] + rot[c][2] * p[2]);
let depth = dist - l[2];
if depth < 0.05 {
return None;
}
let aspect = width as f32 / height as f32;
let ux = (l[0] / depth) / (aspect * 0.5);
let uy = (l[1] / depth) / 0.5;
Some([(ux * 0.5 + 0.5) * width as f32, (1.0 - (uy * 0.5 + 0.5)) * height as f32])
}
fn draw_cross_3d(
painter: &egui::Painter,
clip: egui::Rect,
cross: [f32; 3],
hb: [f32; 3],
rot: &M4,
dist: f32,
width: u32,
height: u32,
ppp: f32,
) {
let painter = painter.with_clip_rect(clip);
let p = [
(cross[0] - 0.5) * 2.0 * hb[0],
(cross[2] - 0.5) * 2.0 * hb[1],
(cross[1] - 0.5) * 2.0 * hb[2],
];
let to_pt = |s: [f32; 2]| egui::Pos2::new(s[0] / ppp, s[1] / ppp);
let color = egui::Color32::from_rgba_unmultiplied(255, 90, 90, 230);
let stroke = egui::Stroke::new(1.5_f32, color);
for axis in 0..3 {
let mut a = p;
let mut b = p;
a[axis] = -hb[axis];
b[axis] = hb[axis];
if let (Some(sa), Some(sb)) = (
project_3d(a, rot, dist, width, height),
project_3d(b, rot, dist, width, height),
) {
painter.line_segment([to_pt(sa), to_pt(sb)], stroke);
}
}
if let Some(c) = project_3d(p, rot, dist, width, height) {
painter.circle_filled(to_pt(c), 4.0, color);
painter.circle_stroke(to_pt(c), 4.0, egui::Stroke::new(1.0_f32, egui::Color32::BLACK));
}
}
fn make_bind(
device: &wgpu::Device,
bgl: &wgpu::BindGroupLayout,
buf: &wgpu::Buffer,
view: &wgpu::TextureView,
sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
device.create_bind_group(&wgpu::BindGroupDescriptor {
label: None,
layout: bgl,
entries: &[
wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view) },
wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
],
})
}
fn make_pipeline(
device: &wgpu::Device,
pl: &wgpu::PipelineLayout,
src: &'static str,
format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
label: None,
source: wgpu::ShaderSource::Wgsl(src.into()),
});
device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
label: None,
layout: Some(pl),
vertex: wgpu::VertexState {
module: &module,
entry_point: Some("vs"),
compilation_options: Default::default(),
buffers: &[],
},
fragment: Some(wgpu::FragmentState {
module: &module,
entry_point: Some("fs"),
compilation_options: Default::default(),
targets: &[Some(format.into())],
}),
primitive: wgpu::PrimitiveState::default(),
depth_stencil: None,
multisample: wgpu::MultisampleState::default(),
multiview: None,
cache: None,
})
}
fn begin_pass<'a>(
enc: &'a mut wgpu::CommandEncoder,
view: &wgpu::TextureView,
clear: bool,
) -> wgpu::RenderPass<'a> {
enc.begin_render_pass(&wgpu::RenderPassDescriptor {
label: None,
color_attachments: &[Some(wgpu::RenderPassColorAttachment {
view,
resolve_target: None,
ops: wgpu::Operations {
load: if clear { wgpu::LoadOp::Clear(wgpu::Color::BLACK) } else { wgpu::LoadOp::Load },
store: wgpu::StoreOp::Store,
},
})],
depth_stencil_attachment: None,
timestamp_writes: None,
occlusion_query_set: None,
})
}
fn render_scene(
pass: &mut wgpu::RenderPass<'_>,
mpr: bool,
w: u32,
h: u32,
p3: &wgpu::RenderPipeline,
p2: &wgpu::RenderPipeline,
b3: &wgpu::BindGroup,
b2: &[wgpu::BindGroup],
) {
if mpr {
let (pw, ph) = (w / 2, h / 2);
let rects = [(0u32, 0u32, pw, ph), (pw, 0, w - pw, ph), (0, ph, pw, h - ph)];
for (i, (x, y, rw, rh)) in rects.iter().enumerate() {
pass.set_viewport(*x as f32, *y as f32, *rw as f32, *rh as f32, 0.0, 1.0);
pass.set_scissor_rect(*x, *y, *rw, *rh);
pass.set_pipeline(p2);
pass.set_bind_group(0, &b2[i], &[]);
pass.draw(0..3, 0..1);
}
} else {
pass.set_pipeline(p3);
pass.set_bind_group(0, b3, &[]);
pass.draw(0..3, 0..1);
}
}
