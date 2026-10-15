# SOTE 3D Dicom viewer

Rust + wgpu alapú 3D DICOM CT-néző Windowsra. Tömörítetlen vagy a `dicom-pixeldata` által dekódolható 16 bites CT-sorozatokat tölt be, és GPU-n volume renderinggel jeleníti meg.

## Fordítás és futtatás

```powershell
cargo run --release -- "D:\DICOM"
```

A végleges önálló `.exe` a `target\release\ct_viewer.exe` útvonalon készül el a `cargo build --release` után.

## Kezelés

- Bal egérgomb: forgatás
- Görgő: zoom
- `1` / `2` / `3`: csont, lágyszövet, tüdő előbeállítás
- `+` / `-`: sűrűség (átlátszóság)

## Korlátok

Nem orvosi eszköz, diagnózisra nem használható.
