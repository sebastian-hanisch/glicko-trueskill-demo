"""Literatur-Referenzwerte für Glicko und TrueSkill (ORACLE-PLAYBOOK §5.1). Nur Werte, die in einer Quelle eingesehen wurden (Stand der Einsicht: 2026-10-17).

Umrechnung: Die Demo rechnet in θ-Einheiten (1 Einheit = SCALE = 400/ln 10 ≈ 173,72 Elo-Punkte). Glicko: µ = (r − 1500)/SCALE, φ = RD/SCALE, c in θ = c in Punkten/SCALE; das
ist Glickmans q = ln 10/400 = 1/SCALE. TrueSkill: Die Demo rechnet mit β = 1,702/√2 statt β = 25/6; die Originaleinheiten x (µ = 25, σ = 25/3, β = 25/6) gehen über den Faktor
SCALE_ORIG = β_Demo/(25/6) in θ-Einheiten, θ = (x − 25)·SCALE_ORIG für µ und θ = σ_x·SCALE_ORIG für σ, zurück x = 25 + θ/SCALE_ORIG bzw. σ_x = σ_θ/SCALE_ORIG.

Werte
-----
1. Glicko-Rechenbeispiel: Spieler 1500 mit RD 200 gegen 1400 (RD 30, Sieg), 1550 (RD 100, Niederlage), 1700 (RD 300, Niederlage).
   Quelle: Glickman, M. E., „The Glicko system“ (Harvard University), http://www.glicko.net/glicko/glicko.pdf, Abschnitt „Example calculation“ (Seite 4), per WebFetch
   abgerufen und im Text der PDF-Seite gelesen am 2026-10-17. Dort: g(RD_j) = 0,9955 / 0,9531 / 0,7242; E = 0,639 / 0,432 / 0,303; d² = 231,67² (= 53670,85); r' = 1500 + 131,9·(−0,272)
   = 1464; RD' = √22918,9 = 151,4. Die Quelle rechnet mit gerundeten Zwischenwerten (d = 231,67 statt 231,70 der ungerundeten Rechnung); verglichen wird mit den Toleranzen der Rundung.
2. Glicko Schritt 1: Zuwachs der Unsicherheit je Periode und Deckel. Quelle: dieselbe PDF, „Step 1 (b)“ und „Implementation issues“ (Seiten 3, 5 und 6): RD = min(√(RD_old² + c²), 350)
   bzw. nach t Perioden ohne Partien RD = min(√(RD_old² + c² t), 350); Beispiel mit RD 50, 30 Perioden, Ziel 350: c = 63,2. Der Wert 63,2 und die Formel stehen in der Quelle.
3. Erwartete Werte von TrueSkill für zwei neue Spieler: Quelle: Dokumentation des Pakets `trueskill`, https://trueskill.org/ (eingesehen am 2026-10-17): Voreinstellung µ = 25, σ = 8,333,
   β = 4,167, τ = 0,0833, Remis-Wahrscheinlichkeit 0,10; `rate_1vs1` zweier neuer Spieler ohne Remis ergibt Sieger Rating(µ=29,396, σ=7,171) und Verlierer Rating(µ=20,604, σ=7,171).
   Das gilt mit der Voreinstellung der Remis-Wahrscheinlichkeit 0,10 (README der Demo: ohne Remis 29,205/7,194). Die Dokumentation nennt auch das Remis-Beispiel (25,000/6,458); die Demo
   kennt keine Remis-Partien und prüft es nicht.
4. Herbrich, R., Minka, T., Graepel, T. (2006): „TrueSkill: A Bayesian Skill Rating System“, NIPS 19 (PDF papers.nips.cc, Seiten als Bilder gelesen am 2026-10-17, weil die Textschicht der PDF unlesbar ist).
   Abschnitt 4.1 (S. 6): Remis-Wahrscheinlichkeit = Φ(ε/(√(n₁+n₂) β)) − Φ(−ε/(√(n₁+n₂) β)) = 2Φ(ε/(√(n₁+n₂) β)) − 1. Tabelle 1 (S. 5), letzte Zeile (Gewinn-Faktor I(x > ε)): mit c := π_x − π_{f→x},
   d := τ_x − τ_{f→x} (kanonische Parameter π = σ⁻², τ = π µ) ist π_x^neu = c/(1 − W(d/√c, ε√c)), τ_x^neu = (d + √c V(d/√c, ε√c))/(1 − W(d/√c, ε√c)); V(t, ε) = N(t − ε)/Φ(t − ε),
   W(t, ε) = V (V + t − ε). Zeilen 3 und 4 der Tabelle: Summe und Differenz gaußscher Größen (Mannschaftsleistung als Summe, Leistungsdifferenz d = t₁ − t₂). Das Papier enthält keine
   nachrechenbaren Zahlenbeispiele für 1 gegen 1 (nur Halo-2-Fehlerraten) und nicht die ausgeschriebene Form µ += σ² v/c; die Zahlen 29,396 usw. stammen aus der Paket-Dokumentation (Punkt 3).
5. Glickman, M. E. (1999): Parameter estimation in large dynamic paired comparison experiments, Appl. Statist. 48 (3), 377–394 (Volltext glicko.net/research/glicko.pdf, gelesen am 2026-10-17).
   Abschnitt 3.3, Gleichungen (9), (10) und (12), (13): die Aktualisierung enthält Elo als Sonderfall: Sind die Gegnerstärken bekannt (σ_j² = 0 für alle Gegner, also g = 1), reduziert sich (9) auf
   die Elo-Aktualisierung µ' = µ + K Σ (s − W_e) mit K = q/(1/σ² + 1/δ²) = q σ'² (Gleichung (13), Punkte-Skala, q = ln 10/400).

Nicht eingesehen oder nicht anwendbar (siehe Bericht „offen“): Elo (1978) (Buch); Glickmans Formel für die erwartete Spielausgang-Wahrscheinlichkeit mit √(RD_i² + RD_j²) (Beispiel 0,376) und die
Match-Qualität des Pakets (44,7 %) hat die Demo nicht umgesetzt; Glicko-2 ist nicht Teil der Demo."""

import math

import pytest

import glt_constants as C
import glt_glicko as G
import glt_trueskill as T
from conftest import build

S = C.SCALE
SCALE_ORIG = C.BETA / (25.0 / 6.0)


def test_glickman_worked_example_weights_expectations_rating_and_deviation():
    games = [(1400.0, 30.0, 1.0), (1550.0, 100.0, 0.0), (1700.0, 300.0, 0.0)]
    for (r_j, rd_j, _), g_paper, e_paper in zip(games, (0.9955, 0.9531, 0.7242), (0.639, 0.432, 0.303)):
        g = G.g_factor(rd_j / S)
        assert g == pytest.approx(g_paper, abs=5e-5)
        assert 1.0 / (1.0 + math.exp(-g * (0.0 - (r_j - 1500.0) / S))) == pytest.approx(e_paper, abs=5e-4)
    mu, phi = G.glicko_update(0.0, 200.0 / S, [((r_j - 1500.0) / S, rd_j / S, s) for r_j, rd_j, s in games])
    assert 1500.0 + S * mu == pytest.approx(1464.0, abs=0.5)                              # Quelle: 1464 (ganzzahlig gerundet)
    assert S * phi == pytest.approx(151.4, abs=0.05)
    # d² der Quelle: 1/RD'² = 1/RD² + 1/d²
    d = 1.0 / math.sqrt(1.0 / (S * phi) ** 2 - 1.0 / 200.0 ** 2)
    assert d == pytest.approx(231.67, abs=0.1)


def test_glickman_step_1_uncertainty_growth_and_the_cap_at_350():
    c = 63.2 / S
    cap = 350.0 / S
    rd = 50.0 / S
    for t in range(1, 31):
        rd = G.inflate(rd, c, cap)                                                        # eine Periode ohne Partien
        assert S * rd == pytest.approx(min(math.sqrt(50.0 ** 2 + 63.2 ** 2 * t), 350.0), abs=1e-9)
    assert S * rd == pytest.approx(350.0, abs=0.5)                                       # nach 30 Perioden fast wieder so unsicher wie ein Unbewerteter
    assert math.sqrt((350.0 ** 2 - 50.0 ** 2) / 30.0) == pytest.approx(63.2, abs=0.05)   # die Herleitung der Quelle
    for _ in range(200):
        rd = G.inflate(rd, c, cap)
    assert S * rd == pytest.approx(350.0, abs=1e-9) and G.inflate(cap, c, cap) == cap     # nie über 350


def _two_new_players(tau_orig: float):
    """Zwei neue Spieler (µ 25, σ 25/3, β 25/6), Remis-Wahrscheinlichkeit 0,10, Spieler 0 gewinnt; Ergebnis in Originaleinheiten."""
    m = build(2, 1, [[(0, 1, True)]])
    run = T.run_trueskill(m, tau_orig * SCALE_ORIG, (25.0 / 3.0) * SCALE_ORIG, draw_probability=0.10, beta=C.BETA)
    return 25.0 + run.mu[0, 0] / SCALE_ORIG, run.sd[0, 0] / SCALE_ORIG, 25.0 + run.mu[0, 1] / SCALE_ORIG, run.sd[0, 1] / SCALE_ORIG


@pytest.mark.parametrize("tau", [0.0, 25.0 / 300.0])
def test_trueskill_documentation_rate_1vs1_of_two_new_players(tau):
    # die Dokumentation rechnet mit τ = 0,0833; das ändert die Werte erst in der fünften Stelle, beide Fassungen runden auf dieselben Zahlen
    mu_w, sd_w, mu_l, sd_l = _two_new_players(tau)
    assert (round(mu_w, 3), round(sd_w, 3)) == (29.396, 7.171)
    assert (round(mu_l, 3), round(sd_l, 3)) == (20.604, 7.171)


# --- Herbrich, Minka, Graepel 2006 (Abschnitt 4.1 und Tabelle 1) -----------------------------------------------------------------------------

@pytest.mark.parametrize("p_draw,beta", [(0.05, 1.0), (0.10, C.BETA), (0.10, 25.0 / 6.0), (0.25, 0.7)])
def test_herbrich_draw_probability_formula_for_one_against_one(p_draw, beta):
    """Abschnitt 4.1: Remis-Wahrscheinlichkeit = 2 Φ(ε/(√(n₁+n₂) β)) − 1 mit n₁ = n₂ = 1."""
    eps = T.draw_margin(p_draw, beta)
    assert 2.0 * T.cdf(eps / (math.sqrt(2.0) * beta)) - 1.0 == pytest.approx(p_draw, abs=1e-9)
    assert T.draw_margin(0.0, beta) == 0.0


def _table_1_win_update(mu_w, sd_w, mu_l, sd_l, beta, eps):
    """Tabelle 1, letzte Zeile, auf die Leistungsdifferenz d = t_W − t_L ~ N(m, c²) angewandt; danach Bedingen der Stärken auf d (Zeilen 3 und 4: gaußsche Summen)."""
    m = mu_w - mu_l
    c2 = 2.0 * beta ** 2 + sd_w ** 2 + sd_l ** 2
    c_, d_ = 1.0 / c2, m / c2                                  # c := π_x − π_{f→x}, d := τ_x − τ_{f→x}; die Nachricht des Faktors ist anfangs 1 (π = τ = 0)
    t, e = d_ / math.sqrt(c_), eps * math.sqrt(c_)
    v = T.pdf(t - e) / T.cdf(t - e)
    w = v * (v + t - e)
    pi_new, tau_new = c_ / (1.0 - w), (d_ + math.sqrt(c_) * v) / (1.0 - w)
    mean_d, var_d = tau_new / pi_new, 1.0 / pi_new
    out = []
    for mu, sd, sign in ((mu_w, sd_w, 1.0), (mu_l, sd_l, -1.0)):
        k = sd ** 2 / c2
        out += [mu + sign * k * (mean_d - m), math.sqrt(sd ** 2 - k * k * (c2 - var_d))]
    return out


@pytest.mark.parametrize("mu_w,sd_w,mu_l,sd_l,beta,eps", [(25.0, 25 / 3, 25.0, 25 / 3, 25 / 6, 0.0), (28.0, 4.0, 22.5, 6.0, 25 / 6, 0.0), (0.4, 0.8, -0.2, 1.3, 0.9, 0.0),
                                                           (0.4, 0.8, -0.2, 1.3, 0.9, 0.35), (-0.5, 1.0, 0.6, 0.5, 1.2, 0.8)])
def test_herbrich_table_1_update_of_a_win(mu_w, sd_w, mu_l, sd_l, beta, eps):
    got = T.trueskill_update(mu_w, sd_w, mu_l, sd_l, beta=beta, eps=eps)
    assert list(got) == pytest.approx(_table_1_win_update(mu_w, sd_w, mu_l, sd_l, beta, eps), rel=1e-10, abs=1e-10)


# --- Glickman 1999, Abschnitt 3.3: Elo als Sonderfall ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mu,phi,games", [(0.0, 200.0 / S, [(100.0, 1.0), (-80.0, 0.0), (250.0, 0.0)]), (0.3, 120.0 / S, [(-150.0, 1.0)]), (-0.2, 340.0 / S, [(0.0, 1.0), (0.0, 0.0), (60.0, 1.0), (-300.0, 1.0)])])
def test_glickman_equation_9_contains_elo_when_the_opponents_are_known(mu, phi, games):
    """Mit φ_j = 0 für alle Gegner ist g = 1 und E die Elo-Erwartung; (9) wird µ' = µ + K Σ (s − W_e) mit K = q σ'² (q = 1 in θ-Einheiten, Gleichung (13))."""
    opp = [((r_j - 1500.0) / S, 0.0, s) for r_j, s in games]
    new_mu, new_phi = G.glicko_update(mu, phi, opp)
    elo_sum = sum(s - 1.0 / (1.0 + math.exp(-(mu - m_j))) for m_j, _, s in opp)
    assert new_mu == pytest.approx(mu + new_phi ** 2 * elo_sum, abs=1e-12)
    assert 1.0 / new_phi ** 2 == pytest.approx(1.0 / phi ** 2 + sum(e * (1.0 - e) for e in (1.0 / (1.0 + math.exp(-(mu - m_j))) for m_j, _, _ in opp)), rel=1e-12)   # Gleichung (10), δ⁻² = q² Σ E (1 − E)
