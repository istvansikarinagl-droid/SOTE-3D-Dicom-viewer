# SOTE 3D Dicom viewer

Rust + wgpu alapú 3D DICOM CT-néző Windowsra. A CT-sorozatot GPU-n volume renderinggel jeleníti meg.

## Mit tud

- DICOM CT-sorozat betöltése mappából
- 3D volume rendering GPU-n (wgpu)
- Csont, lágyszövet és tüdő előbeállítások
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

- Bal egérgomb: forgatás
- Görgő: zoom
- `1` / `2` / `3`: csont, lágyszövet, tüdő előbeállítás
- `+` / `-`: sűrűség (átlátszóság)

## Korlátok

- Nem orvosi eszköz, diagnózisra nem használható.
- Csak 16 bites, a `dicom-pixeldata` által dekódolható DICOM-ot támogat.

## Licenc

MIT licenc, lásd a [LICENSE](LICENSE) fájlt.
