# Tesztelési állapot és ellenőrzési lista

A projekt tulajdonosa a v0.13.0 forráscsomagot helyben lefordította és működőnek jelezte. Ez nem klinikai validáció és nem minden alábbi teszt külön dokumentált eredménye. Új Windows/GPU teszt a dokumentáció készítésekor nem történt.

- [ ] Tiszta klónból cargo build --release sikeres, az ikonfájlok rendelkezésre állnak.
- [ ] Az ablak és a futtatható fájl ikonja megjelenik.
- [ ] A korábbi hasi CT 3D/MPR és W/L nézete működik.
- [ ] A térd MR három megfelelő PD sorozata külön panelre választható.
- [ ] A lapozás és a középső gombos közös pont megfelelően szinkronizál.
- [ ] Eltérő vagy hiányzó FrameOfReferenceUID mellett nincs hamis összekapcsolás.
- [ ] A panelenkénti W/L, zoom és mozgatás működik.
- [ ] A több mérés követi a képet, és csak a saját sorozatán/szeletén látszik.
- [ ] A natív képernyőkép tartalmazza a feliratokat és a méréseket.
- [ ] A 3D figyelmeztetés olvasható és a mentett képen is szerepel.
- [ ] Minimalizálás/visszaállítás és kicsi ablak nem okoz összeomlást.
- [ ] A naplók és betegadatok nem kerülnek a Git-tárolóba.
