"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

import math

SCALE = 400.0 / math.log(10.0)              # Elo-Punkte je logistische Einheit θ (400 Punkte = Faktor 10 der Gewinnchancen), 173,72
BASE = 1500.0                               # Mitte der Elo-Skala in der Anzeige
BETA = 1.702 / math.sqrt(2.0)               # TrueSkill-Leistungsrauschen in θ-Einheiten: Φ(d/(√2 β)) = Φ(d/1,702) ≈ σ(d), damit das Probit-Modell zur logistischen Wahrheit passt
G0 = 10.0                                   # Anlauf-K des Elo-Vergleichs: nach g0 Partien ist der Aufschlag halbiert
BURN = 30                                   # Runden, die beim Verfolgungsfehler nicht zählen (Kaltstart)

N_MIN, N_MAX, DEFAULT_N = 4, 16, 8
DELTA_MIN, DELTA_MAX, DELTA_STEP, DEFAULT_DELTA = 0.05, 1.0, 0.05, 0.3
ROUNDS_MIN, ROUNDS_MAX, DEFAULT_ROUNDS = 20, 200, 150
SIGMA_MIN, SIGMA_MAX, SIGMA_STEP, DEFAULT_SIGMA = 0.0, 0.3, 0.01, 0.06       # Schritt der Irrfahrt je Runde (logistische Einheiten)

SCENARIOS = ("Alle spielen jede Runde", "Ungleiche Aktivität", "Neuling kommt später", "Pause und Rückkehr")
DEFAULT_SCENARIO = SCENARIOS[0]
P_RARE_MIN, P_RARE_MAX, P_RARE_STEP, DEFAULT_P_RARE = 0.05, 0.8, 0.05, 0.2    # Spielwahrscheinlichkeit je Runde der seltenen Hälfte
JOIN_MIN, DEFAULT_JOIN = 2, 60                                                # Runde, in der der Neuling kommt (Obergrenze: Runden − 2)
GAP_MIN, GAP_MAX, GAP_STEP, DEFAULT_GAP = -3.0, 3.0, 0.5, 2.0                 # Abstand des Neulings vom Feld (logistische Einheiten)
PAUSE_START_MIN, DEFAULT_PAUSE_START = 5, 40
PAUSE_LEN_MIN, PAUSE_LEN_MAX, DEFAULT_PAUSE_LEN = 5, 150, 60


def pause_start_max(rounds: int) -> int:
    """Spätester Beginn der Pause: danach bleiben mindestens fünf Runden Pause (plus eine, damit der Längenregler eine Spanne hat) und fünf Runden zum Wiedereinspielen."""
    return rounds - PAUSE_LEN_MIN - 6


def pause_len_max(rounds: int, start: int) -> int:
    """Längste Pause bei gegebenem Beginn: fünf Runden bleiben zum Wiedereinspielen."""
    return min(PAUSE_LEN_MAX, rounds - start - 5)

K_MIN, K_MAX, DEFAULT_K = 1, 200, 28                                          # Elo, in Punkten
PHI0_MIN, PHI0_MAX, DEFAULT_PHI0 = 30, 500, 350                               # Anfangsunsicherheit in Punkten (350 = die übliche Voreinstellung von Glicko)
C_MIN, C_MAX, DEFAULT_C = 1, 60, 8                                            # Zuwachs der Unsicherheit je Runde in Punkten (Glicko c, TrueSkill τ)
SEED_MAX, DEFAULT_SEED = 999999, 7

METHODS = ("Elo", "Glicko", "TrueSkill")

# Feste Instanzen für alle Experimente: nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(100000, 100060))
EXP_N, EXP_DELTA, EXP_ROUNDS = 8, 0.3, 150
SIGMA_SWEEP = (0.0, 0.03, 0.06, 0.12)
K_GRID = (10.0, 20.0, 28.0, 40.0, 56.0, 80.0, 112.0)                          # Punkte
C_GRID = (0.01, 0.02, 0.03, 0.045, 0.06, 0.09, 0.12, 0.18)                    # θ-Einheiten je Runde
PHI0_DEFAULT = 350.0 / SCALE                                                  # 2,01 θ
PHI0_INFORMED = 0.7                                                           # ≈ Standardabweichung der Anfangsstärken des Feldes (n = 8, δ = 0,3: 0,69)
P_RARE_EXP = 0.2
CAL_C = (0.0, 0.01, 0.03, 0.06, 0.12, 0.24)
NEW_JOIN, NEW_ROUNDS, NEW_GAPS = 60, 100, (0.0, 1.0, 2.0)
NEW_GAMES = 20                                                                # Fehler des Neulings nach so vielen eigenen Partien
PAUSE_SIGMAS = (0.06, 0.12)
PAUSE_START, PAUSE_LEN, PAUSE_ROUNDS = 40, 60, 125
PAUSE_GAMES = (1, 5, 20)

COLOR_ELO = "#2F6B65"
COLOR_GLICKO = "#8e6bb0"
COLOR_TRUESKILL = "#d98c1f"
COLOR_TRUTH = "#e67e22"
COLOR_BAND = "rgba(142,107,176,0.20)"
PALETTE = ("#2F6B65", "#8e6bb0", "#d98c1f", "#c0392b", "#3d7ab5", "#7a8f3c", "#b0577e", "#5a5a5a", "#1f9d8a", "#a0522d", "#6c5fc7", "#b8860b",
           "#2e8b57", "#cd5c5c", "#4682b4", "#808000")

_BASE = {"n": DEFAULT_N, "delta": DEFAULT_DELTA, "rounds": DEFAULT_ROUNDS, "sigma": DEFAULT_SIGMA, "scenario": DEFAULT_SCENARIO, "p_rare": DEFAULT_P_RARE,
         "join": DEFAULT_JOIN, "gap": DEFAULT_GAP, "pause_start": DEFAULT_PAUSE_START, "pause_len": DEFAULT_PAUSE_LEN, "k": DEFAULT_K, "phi0": DEFAULT_PHI0,
         "c": DEFAULT_C, "seed": DEFAULT_SEED}
PRESETS = {
    "Eingespieltes Feld": dict(_BASE),
    "Ungleiche Aktivität": {**_BASE, "scenario": SCENARIOS[1], "p_rare": 0.2},
    "Neuling mit Abstand 2": {**_BASE, "scenario": SCENARIOS[2], "sigma": 0.0, "rounds": 100, "join": 60, "gap": 2.0},
    "Lange Pause bei starker Drift": {**_BASE, "scenario": SCENARIOS[3], "sigma": 0.12, "rounds": 150, "pause_start": 40, "pause_len": 60},
}
PRESET_HELP = {
    "Eingespieltes Feld": "8 Spieler, alle spielen jede Runde, die Stärken wandern mit Schritt 0,06: Glicko, TrueSkill und Elo mit gutem K liegen gleichauf.",
    "Ungleiche Aktivität": "Die Hälfte der Spieler spielt nur mit Wahrscheinlichkeit 0,2 je Runde: Wer selten spielt, ist unsicherer; die Voreinstellung 350 Punkte macht ihn zu unsicher.",
    "Neuling mit Abstand 2": "Ein Neuling kommt in Runde 60 mit Startwertung Feldmittel und liegt 2 Einheiten über dem Feld: die Unsicherheit führt ihn schnell hin, festes K braucht lange.",
    "Lange Pause bei starker Drift": "Spieler 0 pausiert 60 Runden, die Stärken wandern stark (0,12 je Runde): Nach der Rückkehr ist seine Wertung veraltet; Glicko wächst in der Unsicherheit mit.",
}
