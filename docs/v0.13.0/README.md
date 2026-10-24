# 3D Dicom viewer — v0.13.0 használati útmutató

Ez a dokumentáció a helyben tesztelt v0.13.0 változatot írja le. A dokumentáció feltöltése önmagában nem jelenti a forráskódfrissítés befejezését. A telepített forrás verzióját a Cargo.toml és a src/native.rs jelenléte alapján is ellenőrizni kell.

## Fordítás

Windows alatt Rust/Cargo MSVC toolchain, Visual Studio C++ Build Tools, Windows SDK és megfelelő grafikus illesztőprogram szükséges.

```powershell
cargo build --release
cargo run --release -- "C:\DICOM"
```

A program: target\release\ct_viewer.exe. Útvonal nélkül mappaválasztó nyílik. Az alkalmazásikonhoz assets/icon.png és assets/icon.ico szükséges; ezeket a bináris fájlokat a szöveges GitHub-összekötő nem tudja feltölteni.

## Több sorozat — natív 2D

Kapcsold be az oldalpanel tetején a Több sorozat — natív 2D módot. A három panel külön sorozatok eredeti képeit mutatja, nem egy térfogatból számolt MPR-metszeteket. Panelenként választhatsz sorozatot és állíthatod a W/L értékeket. A kezdeti választás az orientáció alapján indul, a PD nevű, nem localizer sorozatokat előnyben részesíti; ez heurisztika, felülbírálható.

- Görgő: szeletlapozás az aktív képen.
- Ctrl + görgő: zoom.
- Bal gomb húzása: mozgatás.
- Középső gomb: közös térbeli pont kijelölése vagy húzása.
- Két jobb kattintás: mérés, bekapcsolt natív mérési módban.

A natív képek fizikai pixelaránya megmarad. A mérések sorozathoz és szelethez kötöttek, követik a zoomot és a mozgatást. A mérőszínek: sárga, cián, világoszöld, fehér, ismétlődve.

A térbeli szinkron a közös beteg-koordinátájú ponthoz legközelebbi szeletet választja. Egyező, nem üres FrameOfReferenceUID szükséges; különben az érintett panelek függetlenek. Nincs automatikus regisztráció vagy betegmozgás-korrekció. A piros vonalpár a közös pont vetülete, nem ferde képsíkok pontos metszésvonala. A natív közös pont külön állapot az eredeti MPR/3D kereszthajszáltól.

## Korábbi MPR és 3D

A natív mód kikapcsolásával visszatérhetsz az eredeti nézethez. MPR-ben bal gombbal mozgathatsz, görgővel zoomolhatsz; középső gombbal kereszthajszálat állíthatsz, jobb gombbal mérhetsz. 3D-ben bal gombbal forgathatsz, görgővel közelíthetsz/távolíthatsz. A régi globális W/L nem szabályozza a natív paneleket.

## Képernyőkép

F12 vagy mentési gomb. Mentési párbeszéd kérdezi meg a helyet és nevet; alapnév: DICOM_screenshot_YYYYMMDD_HHMMSS.png. Natív módban a teljes felület mentődik a mérésekkel és feliratokkal. A régi 3D/MPR mentése vezérlőpanelek nélküli. A 3D figyelmeztetés a mentett képre is rákerül.

## Naplók

Az indítási munkakönyvtárban: ct_viewer_load.log, ct_viewer_series.log, ct_viewer_error.log. Ezek útvonalat és sorozatleírást tartalmazhatnak; megosztás előtt ellenőrizd őket. DICOM-vizsgálatot és betegadatot ne tölts fel a forráskód tárolójába.

Nem orvosi eszköz, diagnózisra nem használható. A licencet a tároló LICENSE fájlja tartalmazza; az ikon felhasználási jogát a projekt tulajdonosának kell biztosítania.
