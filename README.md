# SOTE 3D Dicom viewer

Rust + wgpu + egui alapú 3D DICOM CT-néző Windowsra. A CT-sorozatot GPU-n volume renderinggel jeleníti meg, csúszkákkal és menüsorral vezérelhető.

## Mit tud

- DICOM CT-sorozat betöltése mappából
- 3D volume rendering GPU-n (wgpu)
- Csont, erek (kontraszt), szervek és tüdő előbeállítások
- Sima vs. kontrasztanyagos CT váltó
- HU ablak és sűrűség csúszkák
- Három tengely menti metszet sík (X, Y, Z) kapcsolóval és pozíció csúszkával
- Egérrel forgatás, görgővel zoom

## Követelmények

- Rust (rustup)
- Visual Studio Build Tools (Desktop development with C++)

## Fordítás és futtatás

```powershell
cargo run --release -- "D:\DICOM"
```

A végleges önálló `.exe` a `target\release\ct_viewer.exe` útvonalon készül el:

```powershell
cargo build --release
```

## Kezelés

- Bal egérgomb: forgatás (csak ha nem egui panelen vagy)
- Görgő: zoom
- Menüsor: Nézet menüben előbeállítások és kontraszt váltó
- Jobb panel: HU ablak, sűrűség és metszet sík csúszkák

## Korlátok

- Nem orvosi eszköz, diagnózisra nem használható.
- Csak 16 bites, a `dicom-pixeldata` által dekódolható DICOM-ot támogat.
- A kontrasztanyagos mód manuális váltó, nem automatikus felismerés.

## Licenc

MIT licenc, lásd a [LICENSE](LICENSE) fájlt.
