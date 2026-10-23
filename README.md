# 3D Dicom viewer

Rust + wgpu + egui alapú 3D DICOM CT-néző Windowsra.

## Funkciók

- DICOM CT-sorozat betöltése mappából, GPU volume rendering
- Window/Level (szélesség + közép) vezérlés, radiológiai előbeállítások
- Kontrasztanyagos CT mód
- Három tengely menti metszet sík (X, Y, Z)
- MIP (Maximum Intensity Projection) mód
- Izofelület (csont felszín) mód, HU-küszöb csúszkával
- 2D MPR nézet: három panel, panelenkénti zoom és pan
- Kereszthajszál (crosshair), amely összeköti a három panelt
- Távolságmérés jobb egérgombbal, több mérés egyszerre, mm-ben; a mérés a saját szeletéhez kötött
- Képernyőkép mentés mentési dialógussal (F12 vagy gomb), a mérésekkel és feliratokkal együtt
- Hibanaplózás (ct_viewer_error.log) és minimalizálás-védelem

## Kezelés (2D MPR mód)

- Bal egérgomb: kép mozgatása
- Görgő: zoom
- Középső egérgomb: kereszthajszál beállítása (húzható)
- Jobb egérgomb: mérőpont lerakása mérés módban (két klikk = egy mérés), egyébként kereszthajszál
- A mérővonalak a képpel együtt mozognak, és csak a saját szeletükön látszanak
- A mérések színei: sárga, cián, világoszöld, fehér (ciklikusan)

## Képernyőkép

- F12 vagy a „Képernyőkép mentése” gomb
- Mentési dialógus, alapértelmezett név: DICOM_screenshot_YYYYMMDD_HHMMSS.png
- A jelenet a mérésekkel és feliratokkal együtt kerül PNG-be (a UI panelek nélkül)

## Fordítás

```powershell
cargo run --release -- "D:\DICOM"
cargo build --release
```

## Hibakeresés

Panic vagy wgpu hiba esetén a program mappájában a `ct_viewer_error.log` fájl tartalmazza a hibát és a backtrace-t.

## Korlátok

- Nem orvosi eszköz, diagnózisra nem használható.
- Csak 16 bites, a dicom-pixeldata által dekódolható DICOM-ot támogat.
- A mérés nem veszi figyelembe a DICOM képorientációt.

## Licenc

MIT
