# Glicko und TrueSkill: die Zahl mit Unsicherheit – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-glicko-trueskill-demo.streamlit.app/)**

Achtes Stück der **Ranking-Linie** (Konzepte-Demo), Nachfolger von `elo-demo`: Elo schreibt die Wertung nach jeder Partie mit **festem K** fort. Glicko und TrueSkill führen zu
jeder Wertung eine **Unsicherheit** mit, und aus ihr folgt ein K je Spieler und Zeitpunkt. Gemessen wird, wann das etwas bringt, wann nicht und wie gut die Unsicherheit
kalibriert ist. Das Vehikel ist das neutrale, simulierte Turnier der Wurzel (Anfangsstärken aus dem Generator Version 1, wortgleich; danach eine Irrfahrt der Stärken); alles ist
simuliert, es gibt keinen echten Datensatz.

## Kernfrage

Was bringt die mitgeführte Unsicherheit gegenüber einem festen K? **Wenn alle gleich oft spielen, nichts:** Glicko, TrueSkill und Elo mit gut gewähltem K liegen bei jeder Drift
um höchstens 0,015 θ-Einheiten auseinander. Der Gewinn liegt **dort, wo die Information ungleich verteilt ist:** ein **Neuling** mit großem Abstand zum Feld (Fehler nach
20 Partien bei Abstand 2: Glicko 0,38 gegen Elo 1,11 mit festem K und 0,72 mit Anlauf-K), **seltene Spieler** (Gesamtfehler 0,445 gegen 0,474 bei Elo, aber nur mit informiertem Prior), eine **lange Pause**
bei starker Drift (0,43 gegen 0,46, klein). **Die übliche Voreinstellung von 350 Punkten Anfangsunsicherheit schadet** in einem engen Feld (0,500 gegen 0,445 mit informiertem
Prior, Elo 0,474). Und die Unsicherheit ist **nur so gut kalibriert wie der angenommene Zuwachs c:** die Abdeckung des 95-%-Intervalls steigt von 81 % (c = 0) über 97 % (c = 0,06,
der Drift entsprechend) auf 100 % (c = 0,24), während der Fehler wieder wächst.

## Modell

- **Elo:** θ_i += k (S − E), E = σ(θ_i − θ_j), k = K/173,7 (400 Punkte = Faktor 10 der Gewinnchancen); optional Anlauf-K für Neulinge (wie `elo-demo`).
- **Glicko** (Glickman 1999, feste Volatilität, also Glicko-1; die Volatilität von Glicko-2 ist nicht umgesetzt): Wertung µ und Unsicherheit φ (die „Ratings Deviation“). Je Runde
  (eine Bewertungsperiode): φ ← min(√(φ² + c²), φ₀); g(φ_j) = 1/√(1 + 3φ_j²/π²), E = σ(g(φ_j)(µ − µ_j)), 1/v = Σ g² E (1 − E), φ' = 1/√(1/φ² + 1/v), µ' = µ + φ'² Σ g (s − E),
  mit den Werten vom Periodenbeginn für alle Beteiligten.
- **TrueSkill** (Herbrich, Minka, Graepel 2006), 1 gegen 1: vor jeder Partie σ² += τ² für beide; c² = 2β² + σ_w² + σ_l², t = (µ_w − µ_l)/c, v = φ(t − ε/c)/Φ(t − ε/c),
  w = v (v + t − ε/c), µ_w += σ_w² v/c, µ_l −= σ_l² v/c, σ² ← σ² (1 − σ² w/c²). β = 1,702/√2 in θ-Einheiten, damit das Probit-Modell der logistischen Wahrheit entspricht;
  Remis-Marge ε = 0 (die Partien kennen kein Remis).
- **Partien:** Anfangsstärken `make_strengths` (n ≤ 16, Abstand δ); Irrfahrt mit Schritt σ je Runde (Normalverteilung per Box-Muller aus der Gleichverteilten des Generators);
  in jeder Runde spielen die aktiven Spieler in zufälliger Paarung, P(i schlägt j) = σ(θ_i − θ_j). **Szenarien:** alle spielen; die Hälfte spielt nur mit Wahrscheinlichkeit p;
  ein **Neuling** (Spieler 1) kommt in einer Runde mit Startwertung Feldmittel und einem Abstand vom Feld; eine **Pause** von Spieler 1.
- **Maße:** *Verfolgungsfehler* (RMSE der zentrierten Wertungen gegen die zentrierten wahren Stärken ab Runde 30, über die Spieler im Feld, in θ-Einheiten; 1 Einheit = 173,7
  Punkte), *Abdeckung* des 95-%-Intervalls µ ± 1,96 Unsicherheit, *Fehler nach g eigenen Partien* (Neuling, Rückkehrer), *mittleres K* (|Änderung| / |Ergebnis − Erwartung| in
  Punkten).
- Alle Formeln stehen im 📐-Expander der App.

## Methodik

- **Experimente** über 60 feste Läufe (`SWEEP_SEEDS`, nie ein einzelner Seed), je mit einem live berechneten Verdikt; feste Instanzen mit 8 Spielern, Abstand 0,3 und 150 Runden
  (Neuling 100, Pause 125), unabhängig vom Regler. Je Verfahren und Zelle der **beste Abstimmparameter** (K, c oder τ aus einem festen Raster) nach dem Verfolgungsfehler; das ist
  für alle drei optimistisch, weil es die Drift kennt (Grenzen).
- **Aufbau der Vergleiche:** dieselben Partien für alle drei Verfahren; Neuling-Experiment mit festen Parametern (K = 20, c = τ = 0,02), nicht nachgestimmt.
- **Zufall:** `turnier_generator.py` mit SplitMix64-Strom (wortgleich mit den Repos der Linie).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: 60 feste Läufe, 8 Spieler, Abstand 0,3, 150 Runden, wenn nicht anders genannt. Fehler in θ-Einheiten.

| Frage | Befund | Test |
|---|---|---|
| Alle spielen gleich oft, bester Parameter je Verfahren | Drift σ = 0 / 0,03 / 0,06 / 0,12: **Elo 0,239 / 0,269 / 0,334 / 0,486; Glicko 0,229 / 0,270 / 0,339 / 0,483; TrueSkill 0,245 / 0,282 / 0,348 / 0,486.** Abstand Elo–Glicko höchstens 0,010, Elo–TrueSkill höchstens 0,015. | `test_with_equal_activity_the_three_methods_are_level` |
| Bester Parameter folgt der Drift | K = 20 / 20 / 28 / 56 Punkte; c = τ = 0,01 / 0,02 / 0,06 / 0,12. | `test_the_best_parameters_follow_the_drift` |
| Mittleres K je Aktualisierung (σ = 0,06) | Elo 28 (fest), **Glicko 42, TrueSkill 51 Punkte:** die Unsicherheit hält die Verstärkung höher, ohne dass der Fehler sinkt. | `test_the_effective_gain_of_the_methods_with_uncertainty_is_larger_than_elos_fixed_k` |
| Kalibrierung: Abdeckung des 95-%-Intervalls (Glicko), c = 0 / 0,01 / 0,03 / 0,06 / 0,12 / 0,24 | **0,815 / 0,829 / 0,909 / 0,974 / 0,993 / 0,995** (TrueSkill gleich: 0,974 bei c = 0,06). Fehler am kleinsten bei c = 0,06 (0,339) und bei c = 0,24 auf 0,554 gestiegen. | `test_the_coverage_depends_on_c` |
| Ungleiche Aktivität (Hälfte spielt mit p = 0,2), σ = 0,06 | Fehler gesamt: **Elo (K = 40) 0,474; Glicko mit 350 Punkten 0,500; Glicko informiert (0,7 θ) 0,445;** TrueSkill 0,533 und 0,454. Seltene Spieler: Elo 0,518, Glicko informiert 0,476. | `test_the_default_prior_hurts_and_an_informed_prior_helps`, `test_the_rare_players_are_the_hard_ones` |
| Neuling (Fehler nach 20 eigenen Partien, σ = 0) | Abstand 0 / 1 / 2: **Elo fest 0,159 / 0,531 / 1,110; Elo mit Anlauf-K 0,251 / 0,352 / 0,716; Glicko 0,348 / 0,338 / 0,376;** TrueSkill 0,383 / 0,375 / 0,450. | `test_the_newcomer_errors`, `test_the_newcomer_far_from_the_field_is_found_fast_and_the_equal_one_is_made_noisy` |
| Pause (60 Runden), Fehler nach 20 Partien | σ = 0,12: **Elo 0,460, Glicko 0,430 (informiert) bzw. 0,430 (350 Punkte), TrueSkill 0,441;** σ = 0,06: Elo 0,288, Glicko 0,282 (informiert) bzw. 0,288 (350 Punkte); nach der ersten Partie 0,431 gegen 0,454. | `test_the_pause_gain_is_small_and_only_with_strong_drift` |
| Glicko gegen TrueSkill (σ = c = τ = 0,06) | Fehlerdifferenz −0,009 ± 0,011, Kendall τ der Endwertungen 0,987. | `test_glicko_and_trueskill_are_nearly_the_same_method` |

## Befunde und Korrekturen gegenüber dem Plan

- **Korrektur 2026-10-17 (Eigenschaftstest, von Hand gerechnete Handinstanz):** Glicko ließ die Unsicherheit φ in der Pause **nicht** wachsen (Schritt 1 lief nur für Spieler im Feld), obwohl die README das Gegenteil vorsah; das
  Orakel hatte die Regel mitgeschrieben. Mit wachsendem φ ändern sich nur die Pause-Zahlen: σ = 0,06, Fehler nach 20 Partien Glicko informiert 0,303 → 0,282 (Elo 0,288), 350 Punkte 0,320 → 0,288; σ = 0,12 praktisch gleich
  (0,429 → 0,430 bzw. 0,425 → 0,430); nach der ersten Partie bei σ = 0,06 0,454 → 0,431. Die Aussage „klein, bei starker Drift am größten“ bleibt, die Aussage „kein Vorsprung bei schwacher Drift“ entfällt (Vorsprung 0,01 nach 20 Partien).
- **Die Pause bringt weniger als in der Vorab-Messreihe:** 0,384 gegen 0,476 (100 Läufe, σ = 0,12) wurde mit der App-Instanz (60 feste Läufe, Pause in Runde 41 bis 100,
  Messung nach der Rückkehr) zu **0,430 gegen 0,460**. Der Befund steht jetzt als „klein, bei starker Drift am größten“.
- **TrueSkill-Handwert:** Die bekannten Werte 29,396 / 7,171 für zwei Neulinge gelten mit der Remis-Wahrscheinlichkeit 0,10 der Standardeinstellung; **ohne Remis** (wie in dieser
  Demo) sind es **29,205 / 7,194**. Der Plan nannte die erste Zahl für „ohne Remis“; der Orakel-Test prüft beide gegen das Paket `trueskill`.
- **Glicko deckelt die Unsicherheit bei φ₀** (Glickman: RD = min(√(RD² + c²), 350)); der Plan hatte die Formel ohne Deckel. Ohne ihn wächst die Unsicherheit eines Spielers in der
  Pause über die eines Unbewerteten hinaus.
- **TrueSkill-Skala (Phase 0):** β in θ-Einheiten ist 1,702/√2; mit dem Kehrwert lag der Fehler von TrueSkill bei 0,408 statt bei 0,257 ohne Drift (Skalenfehler, kein Befund).
- **Mittleres K statt eigener Kalman-Referenz:** Die Verstärkung der Verfahren mit Unsicherheit liegt im Mittel über der von Elo (42 / 51 gegen 28), ohne dass sich der Fehler
  verbessert; der Plan hatte das nicht erwartet.
- **Anfangsunsicherheit 350 Punkte ist die Voreinstellung der App** (Entscheidung 2); der informierte Prior (0,7 θ-Einheiten = 122 Punkte, die Streuung der Anfangsstärken) steht
  nur in den Experimenten 3 bis 5.

## Abgrenzung

- [Kalman-Filter](https://sebastianhanisch-kalman-filter-demo.streamlit.app/) (Zustandsschätzung und Filter): der Filter mit Varianz; Glicko und TrueSkill sind seine Ranking-Fassung,
  hier ohne Verweis auf die Herleitung.
- **Elo** (Stück 7): festes K; das Anlauf-K dort ist der Behelf für Neulinge, den dieses Stück durch die mitgeführte Unsicherheit ersetzt und gegen ihn misst.
- **Bradley-Terry** (Stück 5): Schätzer über alle Partien; nicht Teil dieses Vergleichs.
- **Nachfolger (nur beim Namen):** WHR und TrueSkill Through Time (rückwärts glätten), Von der Mannschaft zum Einzelnen (Teams).

## Ehrliche Grenzen

- **Gleicher Gewinn wie Elo, wo alle gleich oft spielen** (Experiment 1); der Vorteil der Unsicherheit liegt nur bei ungleicher Information.
- **Zwei Annahmen statt einer:** Anfangsunsicherheit φ₀ und Zuwachs c (τ) stehen nicht in den Daten; die Abdeckung des Intervalls hängt an c (Experiment 2).
- **Die Abstimmung ist optimistisch:** Je Zelle der beste Parameter nach dem Fehler, das setzt die Kenntnis der Drift voraus; in der Praxis ist c eine Annahme.
- **Glicko-1 statt Glicko-2:** keine Volatilität der Stärke. **TrueSkill 1 gegen 1, ohne Remis, ohne Teams.**
- **Eine Partie je Spieler und Runde**, Irrfahrt ohne Rückkehr zum Mittel, n ≤ 16, ein Generator, 60 Läufe je Zelle; Unterschiede unter etwa 0,01 sind Rauschen.
- **Der Neuling-Fehler hängt an den festen Parametern** (K = 20, c = τ = 0,02); andere Werte verschieben die Zahlen.
- **Alles simuliert.**

## Quellen

- Glickman, M. E. (1999): Parameter estimation in large dynamic paired comparison experiments. *Journal of the Royal Statistical Society Series C (Applied Statistics)*
  48 (3), 377–394, doi:10.1111/1467-9876.00159. Formeln, Gewicht g(RD), die Deckelung bei 350 und das Rechenbeispiel (1500 / RD 200 gegen drei Gegner: 1464, RD 151,4) nach dem
  Text „The Glicko system“ von der Seite des Autors (glicko.net), im Test nachgerechnet.
- Herbrich, R., Minka, T., Graepel, T. (2006): TrueSkill™: A Bayesian skill rating system. *Advances in Neural Information Processing Systems 19*, 569–576 (Formeln nach der
  Dokumentation des Pakets `trueskill`; das Original wurde nicht eingesehen). Die Handwerte 29,396 / 7,171 (Remis-Wahrscheinlichkeit 0,10) und 29,205 / 7,194 (ohne Remis) und die
  Aktualisierungen auf Zufallsfolgen sind gegen `trueskill` 0.4.5 geprüft.
- Elo, A. E. (1978): *The Rating of Chessplayers, Past and Present.* Arco Publishing, New York (nach Katalogeinträgen, das Buch wurde nicht eingesehen).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).

## Tests

133 Tests, rund eine Minute: Einheiten gegen die von Hand gerechneten Fälle (`test_elo.py`, `test_glicko.py`, `test_trueskill.py`, `test_drift.py`), Generator
(`test_generator.py`, wortgleich), **Orakel** (`test_oracle_glicko_trueskill.py`: Glicko gegen Glickmans Formeln in Punkten, auch ganze Läufe mit Neuling, Pause und ungleicher
Aktivität; Elo gegen die Lehrbuchformel 10^(ΔR/400); **TrueSkill gegen das Paket `trueskill` 0.4.5 in den Originaleinheiten**, Läufe und einzelne Aktualisierungen mit Remis-Marge),
Maße und Verdikte (`test_evaluation.py`), alle Zahlen dieses README (`test_claims.py`), Presets, ausblendbare Regler (`test_hidden_widget_state.py`), PDF, Laufzeit-Importe und
AppTest-Rauchtests.

**Mutationstest** (`tools/oracle_mutation.py` der Website, eingeschränkt auf die sechs `glt_`-Module): alle **251 Mutationsstellen** einzeln gegen die Einheitentests ohne App-Tests
geprüft: 218 getötet, 33 überlebt. Von den 33 waren **21 echte Lücken** und **12 äquivalent**. Die Lücken lagen in den Randfällen der Maße (eine Runde mit einem einzigen Spieler im
Feld, genau so viele Partien wie gefragt, ein Wert genau auf dem Rand des Intervalls, das Endmaß und die Rangkorrelation der Analyse, die oft spielende Hälfte bei ungerader
Spielerzahl, die Untergrenzen der Pause) und in zwei Verdikten (das beste Verfahren wurde nicht geprüft); sie sind mit Handwert-Tests geschlossen und im zweiten Lauf über die vier
betroffenen Module erneut getötet (184 von 194 getötet, die 10 Überlebenden äquivalent). Die 12 Äquivalenten: Vergleiche bei stetig verteilten Zufallszahlen (`u < p` gegen
`u <= p`), Grenzvergleiche, deren Gleichheitsfall denselben Wert liefert (`x >= 0` im Sigmoid, `pause[1] > pause[0]` bei leerer Pause), Schutzprüfungen gegen eine Division durch null, die nie
greift, und die Wahl der Zentrierungsrunde, die sich in den erlaubten Reglerbereichen nicht unterscheidet.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `turnier_generator.py` | gemeinsamer Generator (wortgleich in allen Repos der Linie; liefert die Anfangsstärken) |
| `glt_drift.py` | Irrfahrt der Stärken, Paarung, Aktivität, Neuling, Pause; `Matches`, `Run` |
| `glt_elo.py` | Elo mit festem K und Anlauf-K |
| `glt_glicko.py` | Glicko: Gewicht, Wachstum der Unsicherheit, Aktualisierung, Lauf |
| `glt_trueskill.py` | TrueSkill 1 gegen 1: v- und w-Funktion, Remis-Marge, Lauf |
| `glt_scenario.py` | Reglerwerte (`Settings`) und die Partien |
| `glt_evaluation.py` | Maße, Analyse, sechs Experimente über die festen Läufe, Verdikte |
| `glt_presets.py`, `glt_constants.py` | Permalink, Presets, ausblendbare Regler, Grenzen |
| `glt_visualization.py`, `glt_pdf_export.py` | Plotly-Abbildungen, PDF-Export |
| `tests/` | Einheiten-, Orakel-, Claims- und App-Tests |

## Bewusst nicht umgesetzt

- **Glicko-2** (Volatilität der Stärke): eigenes iteratives Verfahren, ändert die Kernaussage nicht.
- **Teams, Mehrpersonenspiele, partielles Spiel** (TrueSkill im Faktorgraphen): Stück „Von der Mannschaft zum Einzelnen“.
- **Rückwärts glätten** (WHR, TrueSkill Through Time): eigenes Stück.
- **Remis und Tore im Modell:** die Remis-Marge von TrueSkill ist im Code und im Orakel, in den Partien dieser Demo kommt kein Remis vor.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` (enthält das Paket `trueskill` für das Orakel) und `python -m pytest tests -q`.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html).
