# Porownanie modeli trackingu ROI — Benchmark syntetyczny

## Testowane podejscia (5 PoCow)

| # | Nazwa | Plik | Architektura | Typ |
|---|---|---|---|---|
| 1 | **ViT Improved** | `stabilize_vit_improved.py` | TrackerVit (ONNX) + blokada skali + walidacja histogramu + predykcja velocity | Deep (ViT) |
| 2 | **DaSiamRPN** | `stabilize_dasiamrpn.py` | TrackerDaSiamRPN (ONNX) + blokada skali + walidacja histogramu + velocity | Deep (Siamese) |
| 3 | **Hybrid CSRT+Flow** | `stabilize_hybrid.py` | CSRT + LK optical flow (ROI) + global motion compensation (RANSAC) + Kalman crop | Klasyczny hybryd |
| 4 | **KCF+Flow** | `stabilize_kcf.py` | KCF + LK optical flow fallback + walidacja histogramu | Klasyczny szybki |
| 5 | **Ensemble ViT+CSRT** | `stabilize_ensemble.py` | ViT + CSRT rownolegly, fuzja z votingiem opartym na confidence + velocity | Deep+Klasyczny |

## Scenariusze testowe

| Scenariusz | Co testuje | Klatki | Widocznosc |
|---|---|---|---|
| `normal` | Gladki ruch celu, brak okluzji | 300 | 100% |
| `occlusion` | Cel zostaje zasloniety przechodzacym obiektem, musi powrocic | 300 | 89% |
| `scale_trap` | Reka na tle ciala, okluzja — tracker NIE powinien przeskoczyc na cala postac | 300 | 85% |
| `fast_motion` | Szybki ruch z naglymi zmianami kierunku | 300 | 100% |
| `distractor` | Dwa identyczne obiekty, tracker musi zostac na oryginale | 300 | 100% |
| `face_steal` | **KLUCZOWY**: Reka (cel) przechodzi przed twarza — tracker NIE powinien przeskoczyc na twarz | 300 | 100% |
| `face_steal_occ` | **NAJGORSZY PRZYPADEK**: Reka blisko twarzy + okluzja narzedziem — max confusion | 300 | 90% |

---

## CZESC 1: Wyniki podstawowe (5 scenariuszy)

### Podsumowanie agregowane (5 scenariuszy bazowych)

| Tracker | Sredni blad [px] | Success% | Lost% | Scale Dev | FPS |
|---|---|---|---|---|---|
| **vit_improved** | **2.3** | **92.0%** | 7.5% | 0.044 | 333 |
| dasiamrpn | 29.0 | 86.0% | **0.2%** | 0.130 | 40 |
| hybrid_csrt | 29.9 | 75.6% | 6.4% | 0.027 | 96 |
| kcf_flow | 5.9 | 77.6% | 20.6% | **0.014** | **433** |
| ensemble | 42.2 | 82.5% | 3.1% | 0.035 | 100 |

### Wyniki per scenariusz

#### NORMAL (gladki ruch)

| Tracker | MeanErr | MedErr | Success% | Lost% | FPS |
|---|---|---|---|---|---|
| vit_improved | 4.5 | 2.0 | 88.7% | 10.0% | 324 |
| dasiamrpn | 11.2 | 2.5 | 86.7% | 0.0% | 41 |
| hybrid_csrt | 13.8 | 1.4 | 71.3% | 11.3% | 88 |
| kcf_flow | 7.2 | 5.7 | 65.3% | 32.0% | 420 |
| ensemble | 11.5 | 1.7 | 84.0% | 0.0% | 91 |

**Analiza**: ViT Improved wygrywa pod wzgledem sredniego bledu. KCF traci czesto (32% klatek "lost") bo jest zbyt czuly na zmiane wygladu. DaSiamRPN i Ensemble nigdy nie traca, ale maja wyzszy blad pozycji.

#### OCCLUSION (okluzja + powrot)

| Tracker | MeanErr | Recovery | Success% | Lost% | FPS |
|---|---|---|---|---|---|
| **vit_improved** | **2.3** | **16f** | **87.7%** | 12.0% | 338 |
| dasiamrpn | 127.3 | - | 45.3% | 0.0% | 41 |
| hybrid_csrt | 111.2 | - | 45.0% | 5.0% | 88 |
| **kcf_flow** | **1.5** | **1f** | **95.0%** | 4.7% | 396 |
| ensemble | 195.1 | - | 44.7% | 0.3% | 91 |

**Analiza**: DaSiamRPN, Hybrid i Ensemble **nie traca** trackingu przy okluzji (Lost% bliski 0), ale **sledza zla rzecz** (MeanErr >100px). ViT Improved uczciwie traci tracking (12% lost) ale wraca do wlasciwego celu. KCF+Flow radzi sobie najlepiej — szybko traci i szybko wraca.

#### SCALE TRAP (przeskok na cala postac)

| Tracker | MeanErr | ScaleRatio | ScaleDev | Success% | Lost% |
|---|---|---|---|---|---|
| vit_improved | 1.5 | 0.97 | 0.022 | 84.3% | 15.3% |
| dasiamrpn | 2.8 | **1.17** | **0.286** | 98.7% | 1.0% |
| hybrid_csrt | 7.8 | 1.01 | 0.017 | 84.0% | 15.7% |
| kcf_flow | 1.0 | 1.00 | 0.000 | 84.3% | 15.3% |
| ensemble | 1.3 | 0.98 | 0.025 | 84.3% | 15.3% |

**Analiza**: DaSiamRPN ma ScaleRatio=1.17 i ScaleDev=0.286 — bbox rosnie. Pozostale modele trzymaja skale stabilnie.

#### FAST MOTION (szybki ruch)

| Tracker | MeanErr | Success% | Lost% | FPS |
|---|---|---|---|---|
| **vit_improved** | **1.9** | **99.7%** | 0.0% | 336 |
| **dasiamrpn** | **1.9** | **99.7%** | 0.0% | 40 |
| hybrid_csrt | 15.3 | 78.0% | 0.0% | 82 |
| kcf_flow | 18.1 | 43.7% | **51.0%** | 513 |
| **ensemble** | **1.8** | **99.7%** | 0.0% | 90 |

**Analiza**: Deep trackery doskonale radza sobie z szybkim ruchem. KCF jest katastrofalny (51% lost).

#### DISTRACTOR (podwojny cel)

| Tracker | MeanErr | Success% | FPS |
|---|---|---|---|
| vit_improved | 1.4 | 99.7% | 337 |
| dasiamrpn | 1.7 | 99.7% | 40 |
| hybrid_csrt | 1.5 | 99.7% | 90 |
| kcf_flow | 1.8 | 99.7% | 404 |
| ensemble | **1.1** | **99.7%** | 90 |

---

## CZESC 2: Testy "face steal" — KLUCZOWE DLA SCENARIUSZA CHIRURGICZNEGO

To sa testy ktore bezposrednio adresuja problem: **reka przechodzi przed twarza chirurga/pacjenta
i tracker przeskakuje z reki na twarz**.

### Dlaczego twarz "kradnie" tracking — mechanizm

1. **Feature strength**: Twarz ma silne gradienty (oczy, nos, usta) — w feature space pretrained
   modelu to "kanoniczny obiekt". Reka to gladka, niskoteksturowa plama skory.

2. **Search window overlap**: Kiedy reka jest blisko twarzy, search window trackera zawiera oba obiekty.
   Tracker musi wybrac "ktory region jest bardziej podobny do template". Twarz wygrywa bo:
   - Ma wyzszą aktywacje w warstwach konwolucyjnych (wyuczone na ImageNet/COCO z tysiacami twarzy)
   - Jest stabilna (nie zmienia ksztaltu jak reka)
   - Ma wiekszy contrast z tlem

3. **Online template update**: Po 1-2 klatkach patrzenia na twarz, template zostaje zaaktualizowany
   (w SiamRPN: `res_w = target_sz[0] * (1 - lr) + target[2] * lr`). Tracker "zapomina" o rece.

4. **Brak modelu tozsamosci**: Zadne z podejsc nie wie ze "reka" i "twarz" to rozne kategorie semantyczne.
   Wszystkie operuja na bbox/features, nie na segmentacji.

### Wyniki: face_steal (reka przechodzi przed twarza, BEZ okluzji)

| Tracker | MeanErr | ScaleRatio | ScaleDev | Success% | Lost% | FPS |
|---|---|---|---|---|---|---|
| vit_improved | 8.8 | 1.28 | 0.280 | 41.7% | 57.0% | 328 |
| **dasiamrpn** | **82.3** | **1.58** | **0.395** | 40.0% | 5.3% | 40 |
| hybrid_csrt | 21.2 | 1.03 | 0.224 | 77.0% | 0.0% | 83 |
| kcf_flow | 4.3 | 0.99 | 0.009 | 19.0% | 80.7% | 282 |
| **ensemble** | **4.2** | **1.02** | **0.159** | **99.7%** | **0.0%** | **93** |

**Kluczowe obserwacje:**

- **DaSiamRPN: ScaleRatio=1.58** — bbox rozroslo sie o 58%. Przeskoczyl z reki (65x50) na twarz (90x110).
  MeanErr=82px potwierdza ze sledzi zupelnie inny region.

- **Ensemble: JEDYNY z 99.7% success i err=4.2px** — podwojny tracker z votingiem uchronil sie, bo
  ViT i CSRT oba widzialy reke i sie zgadzaly. Kiedy jeden probiwal przeskoczyc, drugi go korygiowal.

- **ViT Improved: 57% lost** — agresywna walidacja histogramu ODRZUCA twarz (bo wyglada inaczej niz reka),
  ale kosztem ogromnej ilosci "lost" klatek. Chroni sie tracac — bezpieczne ale denerwujace dla uzytkownika.

- **KCF+Flow: 80.7% lost** — traci natychmiast kiedy reka jest blisko twarzy (zbyt wiele textur w okolicy),
  ale kiedy trackuje to trackuje poprawnie (err=4.3px).

- **Hybrid CSRT: 77% success bez lost** — CSRT jest mniej podatny na face steal niz deep trackery,
  bo korelacja CSRT operuje na filtrach przestrzennych a nie na semantycznych features.

### Wyniki: face_steal_occ (reka blisko twarzy + okluzja — NAJGORSZY PRZYPADEK)

| Tracker | MeanErr | Recovery | ScaleDev | Success% | Lost% | FPS |
|---|---|---|---|---|---|---|
| vit_improved | 10.1 | 1f | 0.297 | 40.0% | 59.0% | 327 |
| **dasiamrpn** | **69.0** | 14f | **0.347** | 49.0% | 0.0% | 40 |
| **hybrid_csrt** | **5.3** | **5f** | **0.081** | **87.7%** | **10.0%** | **117** |
| kcf_flow | 8.8 | 8f | 0.102 | 39.3% | 58.3% | 322 |
| **ensemble** | **67.3** | 1f | **0.291** | 54.0% | 10.3% | 126 |

**Kluczowe obserwacje:**

- **Hybrid CSRT wygrywa w najgorszym scenariuszu: 87.7% success, err=5.3px, recovery=5f.**
  Optical flow + RANSAC lepiej radzi sobie z "ktore features sa reki a ktore twarzy" niz deep matching.

- **Ensemble ZAWIODL: err=67.3px, 54% success.** Oba sub-trackery (ViT i CSRT) przeskoczyl na twarz
  po okluzji — voting nie pomaga kiedy obaj sie myla. **To obala hipoteze ze ensemble jest zawsze bezpieczniejszy.**

- **DaSiamRPN: err=69px, 0% lost** — klasyczny problem: "pewnie sledzi zla rzecz". Nie traci nigdy,
  ale po okluzji blisko twarzy przeskakuje na nia natychmiast.

- **ViT Improved: 59% lost, err=10.1 kiedy trackuje** — bezpieczny ale bezuzyteczny (uzytkownik widzi
  zamrozniy obraz przez wiekszosc czasu).

### Podsumowanie face_steal vs ranking bazowy

| Scenariusz | Najlepszy | 2. | 3. | 4. | 5. |
|---|---|---|---|---|---|
| face_steal (bez okluzji) | **Ensemble** (99.7%) | Hybrid (77%) | ViT (41.7%) | DaSiamRPN (40%) | KCF (19%) |
| face_steal_occ (z okluzja) | **Hybrid CSRT** (87.7%) | Ensemble (54%) | DaSiamRPN (49%) | ViT (40%) | KCF (39.3%) |

**Wniosek: Przy face steal ranking sie calkowicie odwraca vs bazowe testy.**

---

## CZESC 3: Ranking ogolny ze wszystkimi 7 scenariuszami

### Ranking po scenariuszu

| Scenariusz | 1. | 2. | 3. | 4. | 5. |
|---|---|---|---|---|---|
| Normal | ViT Improved | KCF+Flow | DaSiamRPN | Ensemble | Hybrid |
| Occlusion | KCF+Flow | ViT Improved | Hybrid | DaSiamRPN | Ensemble |
| Scale Trap | KCF+Flow | Ensemble | ViT Improved | Hybrid | DaSiamRPN |
| Fast Motion | Ensemble | ViT Improved | DaSiamRPN | Hybrid | KCF+Flow |
| Distractor | Ensemble | ViT Improved | Hybrid | DaSiamRPN | KCF+Flow |
| **Face Steal** | **Ensemble** | **Hybrid** | ViT | DaSiamRPN | KCF |
| **Face Steal+Occ** | **Hybrid** | Ensemble | DaSiamRPN | ViT | KCF |

### Ranking ogolny (zaktualizowany)

| Pozycja | Tracker | Uzasadnienie |
|---|---|---|
| **1. Hybrid CSRT+Flow** | Jedyny tracker ktory przezywa face_steal_occ (87.7% success). Przyzwoity w wiekszosci scenariuszy. Optical flow z RANSAC lepiej separuje features reki od twarzy niz deep matching. Umiarkowany FPS (96-117). |
| **2. Ensemble ViT+CSRT** | Najlepszy na face_steal bez okluzji (99.7%!). Dobry na fast motion i distractor. Ale zawodzi na face_steal+occ (voting nie pomaga kiedy obaj sie myla). |
| **3. ViT Improved** | Najlepszy na bazowych testach (92% success, 2.3px error). Chroni sie tracac — bezpieczne ale czesto traci obraz (57-59% lost na face_steal). Najlepszy jesli priorytetem jest "nigdy nie sledzic zlej rzeczy". |
| **4. KCF+Flow** | Ultra-szybki (433 FPS), doskonaly na okluzji generycznej. Ale fatalne face_steal (19-39% success) i fast motion (51% lost). Dobry jako fallback, nie jako primary. |
| **5. DaSiamRPN** | Nigdy nie traci trackingu (0% lost) ale regularnie sledzi zla rzecz po okluzji. Scale drift (1.58 ratio na face steal). Najwolniejszy (40 FPS). Nie nadaje sie do scenariusza chirurgicznego. |

---

## CZESC 4: Dlaczego to sie dzieje i co z tym zrobic

### Fundamentalny problem: bbox tracking nie rozumie tozsamosci obiektu

Wszystkie 5 podejsc dzieli ten sam defekt: **operuja na bounding boxie, nie na masce obiektu**.
Bbox "reka" i bbox "twarz" moga sie nakladac w search window, i tracker nie ma pojecia ze to sa
dwa rozne obiekty. Porownuje features regionu z template i wybiera najsilniejsze dopasowanie.

Twarz jest silniejszym atraktorem niz reka w KAZDEJ architekturze:
- Deep features (ViT, SiamRPN): twarz to kanoniczny obiekt, reka to czesc ciala bez wlasnej tozsamosci
- Korelacja (CSRT, KCF): twarz ma wyraźniejsze gradienty i bardziej stabilna texturę
- Optical flow: features na twarzy (oczy/nos) sa stabilniejsze niz na rece

### Co NIE rozwiaze problemu (na poziomie bbox trackingu)

1. **Wiecej walidacji histogramu** — histogramy reki i twarzy moga byc podobne (oba to skora)
2. **Ostrzejsza blokada skali** — twarz i reka moga byc podobnego rozmiaru
3. **Wiecej trackerow w ensemble** — jesli wszyscy widza twarz jako silniejszy atraktor, voting nie pomoze
4. **Lepsze predykcje velocity** — twarz jest blisko reki, predykcja pozycji nie separuje ich

### Co MOZE rozwiazac problem

1. **Segmentacja masek (SAM2 / Cutie / XMem)**
   - Zamiast bbox, tracker operuje na masce pikseli "reka"
   - Maska reki ma inny ksztalt niz maska twarzy — nawet jesli sie nakladaja, model wie "to nie sa moje piksele"
   - Wymaga GPU, wiecej compute, ale fundamentalnie rozwiazuje problem tozsamosci

2. **Negatywne ROI (exclusion zone)**
   - Uzytkownik zaznacza nie tylko "co sledzic" ale tez "czego unikac"
   - Tracker penalizuje regiony pokrywajace sie z exclusion zone
   - Prostsze niz pelna segmentacja, ale wymaga UX do zaznaczania

3. **Template locking (brak online update)**
   - Zablokowanie aktualizacji template po inicjalizacji
   - Tracker zawsze porownuje z oryginalnym widokiem reki, nigdy nie "uczy sie" twarzy
   - Problem: tracisz odpornosc na zmiane wygladu reki (obrot, deformacja)
   - Kompromis: powolne updates z bardzo niskim learning rate

4. **Spatial prior / motion model**
   - Jesli wiemy ze reka porusza sie w pewien sposob (gora-dol, lewo-prawo), mozemy dac silniejszy prior
   - Twarz jest relatywnie statyczna — jesli tracker nagle "zatrzymuje sie", to podejrzane
   - Implementacja: Kalman filter z modelem ruchu reki, penalizacja za nagla zmiane dynamiki

### Rekomendacja architekturalna (zaktualizowana)

```
Warstwa 1 (Primary):     Hybrid CSRT+Flow
                          - najlepsza odpornosc na face steal z okluzja
                          - RANSAC separuje features lepiej niz deep matching

Warstwa 2 (Verification): Template lock + histogram distance
                          - zamroz oryginalny template
                          - jesli distance od template rosnie > prog, wlacz ostrzezenie

Warstwa 3 (Recovery):     ViT Improved
                          - uzywaj jako "re-detector" co N klatek
                          - ale TYLKO jesli wykryty region przechodzi walidacje skali + appearance

Warstwa 4 (Fallback):     Bezpieczny tryb
                          - zamroz obraz + zoom out + "REACQUIRING"
                          - NIE probuj sledzic nic jesli confidence < prog

Przyszlosc:               SAM2 mask tracking
                          - jedyne rozwiazanie ktore fundamentalnie
                            rozwiaze problem "reka vs twarz"
```

---

## Srodowisko testowe

- OpenCV 4.13.0 (contrib)
- Python 3.14
- Syntetyczne klatki 640x480, 300 klatek per scenariusz (7 scenariuszy, 2100 klatek total)
- Metryki: center error (px), scale ratio, scale deviation, lost%, success rate (<30px), recovery time, FPS
- Platforma: Linux aarch64
- Benchmark: `benchmark.py` — deterministyczne sekwencje z kontrolowana okluzja, twarz, reka, distraktor
