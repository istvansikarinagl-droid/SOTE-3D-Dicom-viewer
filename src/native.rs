
use super::*;

struct Stack {
    name: String,
    uid: String,
    frame: String,
    slices: Vec<Slice>,
    normal: [f64; 3],
    lo: f32,
    hi: f32,
    axis: usize,
}
struct Mark {
    series: usize,
    slice: usize,
    points: [[f32; 2]; 2],
    mm: f64,
    color: egui::Color32,
}
struct Panel {
    series: usize,
    slice: usize,
    zoom: f32,
    pan: egui::Vec2,
    width: f32,
    level: f32,
    texture: Option<egui::TextureHandle>,
    key: Option<(usize, usize, u32, u32)>,
}
pub(super) struct NativeViewer {
    stacks: Vec<Stack>,
    panels: Vec<Panel>,
    point: [f64; 3],
    marks: Vec<Mark>,
    pending: Option<(usize, usize, [f32; 2])>,
    measure: bool,
    sync: bool,
    source: usize,
    status: String,
}
fn unit(v: [f64; 3]) -> Option<[f64; 3]> {
    let l = dot(v,v).sqrt();
    if !l.is_finite() || l < 1e-6 { return None; }
    Some(v.map(|x| x/l))
}
fn basis(s: &Slice) -> ([f64; 3], [f64; 3]) {
    ([s.iop[0],s.iop[1],s.iop[2]], [s.iop[3],s.iop[4],s.iop[5]])
}
fn patient(s: &Slice, uv: [f32; 2]) -> [f64; 3] {
    let (u,v) = basis(s);
    let x = uv[0] as f64 * s.cols as f64 - 0.5;
    let y = uv[1] as f64 * s.rows as f64 - 0.5;
    [0,1,2].map(|k| s.pos[k] + u[k]*x*s.sx + v[k]*y*s.sy)
}
fn image_point(s: &Slice, p: [f64; 3]) -> [f32; 2] {
    let (u,v)=basis(s);
    let d=[0,1,2].map(|k| p[k]-s.pos[k]);
    [((dot(d,u)/s.sx+0.5)/s.cols as f64) as f32,
     ((dot(d,v)/s.sy+0.5)/s.rows as f64) as f32]
}
fn native_pos(uv: [f32;2], rect: egui::Rect) -> egui::Pos2 {
    egui::pos2(rect.left()+uv[0]*rect.width(), rect.top()+uv[1]*rect.height())
}
impl NativeViewer {
    pub(super) fn load(dir: &Path) -> Self {
        let mut files=Vec::new(); collect_files(dir,&mut files); files.sort();
        let mut groups: HashMap<String, (String,String,Vec<Slice>)> = HashMap::new();
        let mut notes=Vec::new();
        for path in files {
            let Ok(obj)=open_file(&path) else {continue;};
            let geom=obj.element(tags::IMAGE_ORIENTATION_PATIENT).ok()
                .and_then(|e|e.to_multi_float64().ok());
            if geom.as_ref().map_or(true, |v|v.len()!=6 || v.iter().any(|x|!x.is_finite())) {continue;}
            let frame=obj.element(tags::FRAME_OF_REFERENCE_UID).ok()
                .and_then(|e|e.to_str().ok().map(|s|s.trim().to_owned())).unwrap_or_default();
            let spacing=obj.element(tags::PIXEL_SPACING).ok().and_then(|e|e.to_multi_float64().ok());
            if spacing.as_ref().map_or(true, |v| v.len()!=2 || v.iter().any(|x| !x.is_finite() || *x<=0.0)) {continue;}
            let samples=obj.element(tags::SAMPLES_PER_PIXEL).ok().and_then(|e|e.to_int::<u16>().ok()).unwrap_or(1);
            let bits=obj.element(tags::BITS_ALLOCATED).ok().and_then(|e|e.to_int::<u16>().ok()).unwrap_or(0);
            if samples!=1 || bits!=16 {continue;}
            let Ok((uid,sl))=read_slice(&path) else {continue;};
            if sl.rows==0 || sl.cols==0 || sl.sx<=0.0 || sl.sy<=0.0 || !sl.sx.is_finite() || !sl.sy.is_finite()
                || sl.pos.iter().any(|x|!x.is_finite()) {continue;}
            let (u,v)=basis(&sl);
            if (dot(u,u)-1.0).abs()>0.01 || (dot(v,v)-1.0).abs()>0.01 || dot(u,v).abs()>0.01 {continue;}
            let orient=sl.iop.map(|x|(x*1000.0).round() as i32);
            let key=format!("{}|{}|{:?}|{}|{}|{:.5}|{:.5}",uid,frame,orient,sl.cols,sl.rows,sl.sx,sl.sy);
            groups.entry(key).or_insert_with(||(uid,frame,Vec::new())).2.push(sl);
        }
        let mut stacks=Vec::new();
        for (_, (uid,frame,mut slices)) in groups {
            let (u,v)=basis(&slices[0]);
            let Some(normal)=unit(cross(u,v)) else {continue;};
            slices.sort_by(|a,b|dot(a.pos,normal).total_cmp(&dot(b.pos,normal)));
            let mut unique=Vec::new();
            for sl in slices {
                if unique.last().map_or(false, |last:&Slice| (dot(sl.pos,normal)-dot(last.pos,normal)).abs()<0.001) {
                    notes.push(format!("Azonos helyű kép kihagyva: {} (idő/echo dimenzió még nem támogatott)",uid));
                    continue;
                }
                unique.push(sl);
            }
            let mut sample:Vec<f32>=unique.iter().flat_map(|s|s.px.iter().step_by(16).copied()).filter(|x|x.is_finite()).collect();
            if sample.is_empty() {continue;}
            sample.sort_by(|a,b|a.total_cmp(b));
            let lo=sample[0]; let hi=sample[((sample.len()-1) as f64*0.995) as usize].max(lo+1.0);
            let axis=dominant(normal).0;
            let plane=["SAG","COR","AXI"][axis];
            let name=format!("{} | {} | {} kép | {}×{}",unique[0].description,plane,unique.len(),unique[0].cols,unique[0].rows);
            notes.push(format!("{}; FrameOfReferenceUID={}",name,frame));
            stacks.push(Stack{name,uid,frame,slices:unique,normal,lo,hi,axis});
        }
        stacks.sort_by(|a,b|a.name.cmp(&b.name).then(a.uid.cmp(&b.uid)));
        let mut panels=Vec::new();
        for axis in [0,1,2] {
            let best=stacks.iter().enumerate().filter(|(_,s)|s.axis==axis).max_by_key(|(_,s)| {
                let n=s.name.to_lowercase();
                (if n.contains("localizer") || n.contains("scout") {0} else if n.contains("pd") {3} else {2},s.slices.len())
            }).map(|(i,_)|i).unwrap_or(0);
            if let Some(st)=stacks.get(best) {
                panels.push(Panel{series:best,slice:st.slices.len()/2,zoom:1.0,pan:egui::Vec2::ZERO,
                    width:st.hi-st.lo,level:(st.hi+st.lo)*0.5,texture:None,key:None});
            }
        }
        let point=panels.first().map(|p|patient(&stacks[p.series].slices[p.slice],[0.5,0.5])).unwrap_or([0.0;3]);
        let _=std::fs::write("ct_viewer_series.log",notes.join("\n"));
        let mut out=Self{stacks,panels,point,marks:Vec::new(),pending:None,measure:false,sync:true,source:0,status:String::new()};
        out.synchronize(0); out
    }
    fn compatible(&self,a:usize,b:usize)->bool {
        let x=&self.stacks[self.panels[a].series]; let y=&self.stacks[self.panels[b].series];
        !x.frame.is_empty() && x.frame==y.frame
    }
    fn synchronize(&mut self, source:usize) {
        if self.panels.is_empty() {return;}
        self.source=source;
        self.status.clear();
        if !self.sync {return;}
        for i in 0..self.panels.len() {
            if i==source {continue;}
            if !self.compatible(source,i) {
                self.status="Hiányzó / eltérő FrameOfReferenceUID: az érintett panelek lapozása független.".into(); continue;
            }
            let p=&mut self.panels[i]; let st=&self.stacks[p.series];
            p.slice=st.slices.iter().enumerate().min_by(|(_,a),(_,b)| {
                let da=dot([0,1,2].map(|k|self.point[k]-a.pos[k]),st.normal).abs();
                let db=dot([0,1,2].map(|k|self.point[k]-b.pos[k]),st.normal).abs(); da.total_cmp(&db)
            }).map(|(j,_)|j).unwrap_or(0);
        }
    }
    pub(super) fn controls(&mut self,ui:&mut egui::Ui) {
        if ui.checkbox(&mut self.sync,"Térbeli szinkronizálás").changed() {self.synchronize(self.source);}
        if self.panels.is_empty() {ui.colored_label(egui::Color32::YELLOW,"Nincs támogatott natív sorozat (egykockás, 16 bites, érvényes geometriájú képek szükségesek).");}
        ui.checkbox(&mut self.measure,"Jobb klikkes mérés (natív képek)");
        if ui.button("Natív mérések törlése").clicked() {self.marks.clear(); self.pending=None;}
        ui.label("Görgő: szelet • Ctrl+görgő: zoom");
        ui.label("Bal húzás: mozgatás • Középső: közös pont");
        ui.label("A szinkron nem végez mozgáskorrekciót / regisztrációt.");
        if !self.status.is_empty() {ui.colored_label(egui::Color32::YELLOW,&self.status);}
    }
    pub(super) fn show(&mut self,ctx:&egui::Context,area:egui::Rect) {
        if self.panels.is_empty() {return;}
        let mut changed=None;
        for i in 0..self.panels.len() {
            let size=egui::vec2(area.width()/3.0,area.height());
            let origin=area.min+egui::vec2(i as f32*size.x,0.0);
            egui::Area::new(egui::Id::new(("native_panel",i))).order(egui::Order::Middle)
                .fixed_pos(origin).movable(false).show(ctx,|ui| {
                ui.set_min_size(size); ui.set_max_size(size);
                let linked=i==self.source || (self.sync && self.compatible(self.source,i));
                let p=&mut self.panels[i]; let old=p.series;
                egui::ComboBox::from_id_salt(("series_select",i)).width((size.x-20.0).max(20.0))
                    .selected_text(&self.stacks[p.series].name).show_ui(ui,|ui| {
                    for (j,st) in self.stacks.iter().enumerate() {ui.selectable_value(&mut p.series,j,&st.name);}
                });
                if old!=p.series {
                    let st=&self.stacks[p.series]; p.slice=st.slices.len()/2; p.zoom=1.0; p.pan=egui::Vec2::ZERO;
                    p.width=st.hi-st.lo;p.level=(st.hi+st.lo)/2.0;self.pending=None;
                    self.point=patient(&st.slices[p.slice],[0.5,0.5]);changed=Some(i);
                }
                let st=&self.stacks[p.series];
                let before=p.slice;
                ui.add(egui::Slider::new(&mut p.slice,0..=st.slices.len()-1).text("Szelet"));
                ui.add(egui::Slider::new(&mut p.width,1.0..=(st.hi-st.lo).max(2.0)*2.0).text("W"));
                ui.add(egui::Slider::new(&mut p.level,st.lo..=st.hi).text("L"));
                let height=(size.y-150.0).max(20.0);
                let (view,response)=ui.allocate_exact_size(egui::vec2((size.x-8.0).max(10.0),height),egui::Sense::click_and_drag());
                let painter=ui.painter().with_clip_rect(view);
                painter.rect_filled(view,0.0,egui::Color32::BLACK);
                if response.dragged_by(egui::PointerButton::Primary) {p.pan+=ctx.input(|x|x.pointer.delta());}
                if response.hovered() {
                    let scroll=ctx.input(|x|x.raw_scroll_delta.y);
                    if scroll!=0.0 {
                        if ctx.input(|x|x.modifiers.ctrl) {p.zoom=(p.zoom*(scroll*0.002).exp()).clamp(0.2,20.0);}
                        else {p.slice=(p.slice as isize+if scroll>0.0 {-1} else {1}).clamp(0,st.slices.len() as isize-1) as usize;}
                    }
                }
                let sl=&st.slices[p.slice];
                if before!=p.slice {
                    let delta=dot([0,1,2].map(|k|sl.pos[k]-self.point[k]),st.normal);
                    for k in 0..3 {self.point[k]+=delta*st.normal[k];} changed=Some(i);
                    self.pending=None;
                }
                let physical=egui::vec2(sl.cols as f32*sl.sx as f32,sl.rows as f32*sl.sy as f32);
                let scale=(view.width()/physical.x).min(view.height()/physical.y)*p.zoom;
                let image_rect=egui::Rect::from_center_size(view.center()+p.pan,physical*scale);
                let key=(p.series,p.slice,p.width.to_bits(),p.level.to_bits());
                if p.key!=Some(key) {
                    let low=p.level-p.width*0.5;
                    let rgba:Vec<u8>=sl.px.iter().flat_map(|&x| {
                        let v=(((x-low)/p.width).clamp(0.0,1.0)*255.0).round() as u8; [v,v,v,255]
                    }).collect();
                    let img=egui::ColorImage::from_rgba_unmultiplied([sl.cols as usize,sl.rows as usize],&rgba);
                    if let Some(t)=&mut p.texture {t.set(img,egui::TextureOptions::LINEAR);} else {
                        p.texture=Some(ctx.load_texture(format!("native_{i}"),img,egui::TextureOptions::LINEAR));
                    } p.key=Some(key);
                }
                if let Some(t)=&p.texture {painter.image(t.id(),image_rect,egui::Rect::from_min_max(egui::Pos2::ZERO,egui::pos2(1.0,1.0)),egui::Color32::WHITE);}
                let uv_at=|pos:egui::Pos2| [(pos.x-image_rect.left())/image_rect.width(),(pos.y-image_rect.top())/image_rect.height()];
                if response.clicked_by(egui::PointerButton::Middle) || response.dragged_by(egui::PointerButton::Middle) {
                    if let Some(pos)=response.interact_pointer_pos() {
                        let uv=uv_at(pos); if uv.iter().all(|x| *x>=0.0 && *x<=1.0) {self.point=patient(sl,uv); changed=Some(i);}
                    }
                }
                if self.measure && response.clicked_by(egui::PointerButton::Secondary) {
                    if let Some(pos)=response.interact_pointer_pos() {
                        let uv=uv_at(pos);
                        if uv.iter().all(|x|*x>=0.0 && *x<=1.0) {
                            match self.pending.take() {
                                Some((series,slice,a)) if series==p.series && slice==p.slice => {
                                    let dx=(a[0]-uv[0]) as f64*sl.cols as f64*sl.sx;
                                    let dy=(a[1]-uv[1]) as f64*sl.rows as f64*sl.sy;
                                    self.marks.push(Mark{series,slice,points:[a,uv],mm:(dx*dx+dy*dy).sqrt(),color:MEASURE_COLORS[self.marks.len()%4]});
                                }
                                _=>self.pending=Some((p.series,p.slice,uv)),
                            }
                        }
                    }
                }
                if linked {
                let c=native_pos(image_point(sl,self.point),image_rect);
                let stroke=egui::Stroke::new(1.0,egui::Color32::LIGHT_RED);
                painter.line_segment([egui::pos2(c.x,view.top()),egui::pos2(c.x,view.bottom())],stroke);
                painter.line_segment([egui::pos2(view.left(),c.y),egui::pos2(view.right(),c.y)],stroke);
                }
                for m in &self.marks {
                    if m.series!=p.series || m.slice!=p.slice {continue;}
                    let a=native_pos(m.points[0],image_rect);let b=native_pos(m.points[1],image_rect);
                    painter.line_segment([a,b],egui::Stroke::new(2.0,m.color));
                    painter.circle_filled(a,3.0,m.color);painter.circle_filled(b,3.0,m.color);
                    painter.text(a+(b-a)*0.5,egui::Align2::CENTER_BOTTOM,format!("{:.1} mm",m.mm),egui::FontId::proportional(16.0),m.color);
                }
                if let Some((series,slice,a))=self.pending {if series==p.series && slice==p.slice {painter.circle_filled(native_pos(a,image_rect),4.0,egui::Color32::YELLOW);}}
            });
        }
        if let Some(i)=changed {self.synchronize(i);ctx.request_repaint();}
    }
}
pub(super) fn warning(vol:&Volume)->String {
    let min=vol.spacing.iter().copied().fold(f64::INFINITY,f64::min);
    let max=vol.spacing.iter().copied().fold(0.0,f64::max);
    if !min.is_finite() || min<=0.0 {return "FIGYELEM: hiányos geometria. A 3D méretarány nem megbízható.".into();}
    if max/min>3.0 {
        format!("FIGYELEM — KORLÁTOZOTT 3D RÉSZLETESSÉG\nVoxelméret: {:.3} × {:.3} × {:.3} mm; anizotrópia: {:.1}×\nA rekonstrukció nem pótolja a hiányzó részleteket. Használd a natív 2D sorozatokat.",vol.spacing[0],vol.spacing[1],vol.spacing[2],max/min)
    } else {String::new()}
}
pub(super) fn draw_warning(p:&egui::Painter,rect:egui::Rect,text:&str) {
    if text.is_empty() {return;}
    let font=egui::FontId::proportional(16.0);
    let galley=p.layout(text.into(),font,egui::Color32::YELLOW,(rect.width()-24.0).max(40.0));
    let pos=rect.min+egui::vec2(12.0,12.0);
    p.rect_filled(egui::Rect::from_min_size(pos-egui::vec2(6.0,6.0),galley.size()+egui::vec2(12.0,12.0)),4.0,egui::Color32::from_black_alpha(230));
    p.galley(pos,galley,egui::Color32::YELLOW);
}

