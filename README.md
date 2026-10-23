# 3D Dicom viewer

Rust + wgpu + egui alapú 3D DICOM CT-néző Windowsra.

## Funkciók

- DICOM CT-sorozat betöltése mappából, GPU volume rendering
- Window/Level (szélesség + közép) vezérlés, radiológiai előbeállítások
- Kontrasztanyagos CT mód
- Három tengely menti metszet sík (X, Y, Z)
- MIP (Maximum Intensity Projection) mód
- Izofelület (csont felszín) mód, HU-küszöb csúszkával
- 2D MPR nézet: axiális, szagittális, koronális, panelenkénti zoom és pan
- Távolságmérés jobb egérgombbal, több mérés egyszerre, mm-ben
- Képernyőkép mentés mentési dialógussal (F12 vagy gomb)

## Kezelés (2D MPR mód)

- Bal egérgomb: kép mozgatása
- Görgő: zoom
- Jobb egérgomb: mérőpont lerakása (két klikk = egy mérés)
- A mérővonalak a képpel együtt mozognak zoomoláskor és mozgatáskor is
- A mérések színei: sárga, cián, világoszöld, fehér (ciklikusan)

## Képernyőkép

- F12 vagy a „Képernyőkép mentése” gomb
- Mentési dialógus, alapértelmezett név: DICOM_screenshot_YYYYMMDD_HHMMSS.png
- A jelenet UI nélkül kerül mentésre PNG formátumban

## Fordítás

```powershell
cargo run --release -- "D:\DICOM"
cargo build --release
```

## Korlátok

- Nem orvosi eszköz, diagnózisra nem használható.
- Csak 16 bites, a dicom-pixeldata által dekódolható DICOM-ot támogat.

## Licenc

MIT