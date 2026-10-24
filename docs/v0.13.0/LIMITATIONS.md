# Ismert korlátok

- Natív bemenet: egykockás, 16 bites, egycsatornás DICOM, érvényes pozícióval, orientációval és pixelmérettel.
- Enhanced többkockás DICOM teljes geometriai feldolgozása nincs implementálva.
- Több echo/időpont teljes szétválasztása nem támogatott; azonos helyen ismétlődő képekből az első marad meg, a kihagyást naplózzuk.
- A natív csoportosítás sorozat UID, FrameOfReferenceUID, orientáció, méret és pixelméret alapján fut.
- A natív nézet megtartja az eredeti síkot. A régi 3D betöltő domináns tengelyek szerinti permutációt/tükrözést végez, nem teljes ferde újramintavételezést. Egyenetlen szeletközöknél medián térközt használ.
- A régi 3D betöltő a legtöbb képes sorozatot választja. A natív sorozatválasztó nem választ új 3D térfogatot.
- Az automatikus CT/MR felülbírálás elsősorban megjelenítési beállításokat cserél; a betöltéskor levágott intenzitások módváltással nem állíthatók vissza.
- A küszöbös MR 3D megjelenítés nem csont-, porc- vagy érszegmentáció.
- A natív modul újraolvassa és memóriában tartja a sorozatokat; nagy vizsgálatoknál magasabb memóriaigény és lassabb indulás várható.
- A winit ApplicationHandler átállás nincs benne, deprecated figyelmeztetések maradhatnak.

## 3D figyelmeztetés

A kiválasztott térfogat legnagyobb/kisebb voxelméretének aránya >3 esetén sárga figyelmeztetés jelenik meg. A 3-as küszöb fejlesztői heurisztika, nem klinikai minőségi határ. A betöltési napló geometriai figyelmeztetései is megjelenhetnek. A felirat hiánya nem igazolja a 3D rekonstrukció megfelelő minőségét. A program nem állítja a sorozatnevekből, hogy a teljes vizsgálatban nincs izotróp 3D sorozat.

## Szinkron és mérés

Azonos FrameOfReferenceUID sem garantál betegmozgás nélküli illeszkedést. Nincs képalapú regisztráció. Eltérő képfedettségnél a közös pont a képen kívül lehet. A mm-kijelzés nem hitelesített orvosi mérés.
