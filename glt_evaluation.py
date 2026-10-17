"""Kennzahlen, Analyse des gezeigten Laufs, sechs Experimente über die festen Läufe (SWEEP_SEEDS) und die Verdikte.

Alle drei Verfahren sehen dieselben Partien. Fehlermaße in θ-Einheiten (1 Einheit = 173,7 Elo-Punkte); Verfolgungsfehler = RMSE der zentrierten Wertungen gegen die
zentrierten wahren Stärken ab Runde BURN, über die Spieler im Feld; Abdeckung = Anteil der (Runde, Spieler)-Paare, in denen die wahre Stärke im Intervall µ ± 1,96 σ liegt
(beide Seiten über die Spieler im Feld zentriert)."""

import functools
import math
from dataclasses import dataclass

import numpy as np

import glt_constants as C
import glt_drift as D
import glt_elo as E
import glt_glicko as G
import glt_trueskill as T
from glt_scenario import Settings, make_matches

NAN = float("nan")


def de(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float, digits: int = 0) -> str:
    return de(100 * x, digits)


def _mean(values) -> float:
    v = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
    return float(sum(v) / len(v)) if v else NAN


# --- Maße -------------------------------------------------------------------------------------------------------------------------------------

def _present(m: D.Matches, t: int) -> np.ndarray:
    return m.presence[t]


def _round_stats(run: D.Run, m: D.Matches, subset=None):
    """Je Runde: Zahl der Spieler im Feld, zentrierte Abweichung (Wertung − Stärke, beide über die Spieler im Feld zentriert) und die Maske der Spieler im Feld."""
    ok = m.presence.copy()
    if subset is not None:
        mask = np.zeros(m.n, dtype=bool)
        mask[list(subset)] = True
        ok &= mask[None, :]
    cnt = ok.sum(axis=1)
    safe = np.maximum(cnt, 1)
    mu = np.where(ok, run.mu, 0.0)
    tr = np.where(ok, m.truth, 0.0)
    d = np.where(ok, (mu - (mu.sum(axis=1) / safe)[:, None]) - (tr - (tr.sum(axis=1) / safe)[:, None]), 0.0)
    return cnt, d, ok


def round_error(run: D.Run, m: D.Matches, t: int, subset=None) -> float:
    """RMSE der zentrierten Wertungen gegen die zentrierten Stärken in Runde t über die Spieler im Feld (optional nur über `subset`, zentriert in der Teilmenge)."""
    cnt, d, _ = _round_stats(run, m, subset)
    return float(math.sqrt((d[t] ** 2).sum() / cnt[t])) if cnt[t] >= 2 else NAN


def tracking_error(run: D.Run, m: D.Matches, burn: int = C.BURN, subset=None) -> float:
    cnt, d, _ = _round_stats(run, m, subset)
    use = cnt[burn:] >= 2
    if not use.any():
        return NAN
    rmse = np.sqrt((d[burn:] ** 2).sum(axis=1) / np.maximum(cnt[burn:], 1))
    return float(rmse[use].mean())


def coverage(run: D.Run, m: D.Matches, burn: int = C.BURN) -> float:
    """Abdeckung des 95-%-Intervalls (nur Verfahren mit Unsicherheit); NaN bei Elo."""
    if run.sd is None:
        return NAN
    cnt, d, ok = _round_stats(run, m)
    use = (cnt >= 2)[:, None] & ok
    use[:burn] = False
    sd = np.where(ok, run.sd, 0.0)
    hit = (np.abs(d) <= 1.96 * sd) & use
    tot = int(use.sum())
    return float(hit.sum() / tot) if tot else NAN


def player_rounds(m: D.Matches, player: int, after: int = 0) -> list:
    """Runden ab `after`, in denen der Spieler eine Partie hatte (aufsteigend)."""
    seen = sorted({int(r) for r, i, j in zip(m.rnd, m.i, m.j) if (i == player or j == player) and r >= after})
    return seen


def player_error(run: D.Run, m: D.Matches, t: int, player: int = 0) -> float:
    """|zentrierte Wertung − zentrierte wahre Stärke| des Spielers in Runde t (zentriert über die Spieler im Feld)."""
    ok = _present(m, t)
    r, s = run.mu[t], m.truth[t]
    return float(abs((r[player] - r[ok].mean()) - (s[player] - s[ok].mean())))


def error_after_games(run: D.Run, m: D.Matches, games: int, after: int = 0) -> float:
    """Fehler von Spieler 0 in der Runde seiner `games`-ten Partie (gezählt ab Runde `after`); NaN, wenn es so viele Partien nicht gibt."""
    rounds = player_rounds(m, 0, after)
    if len(rounds) < games:
        return NAN
    return player_error(run, m, rounds[games - 1], 0)


def kendall_tau(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    iu = np.triu_indices(len(a), 1)
    sa, sb = np.sign(a[iu[0]] - a[iu[1]]), np.sign(b[iu[0]] - b[iu[1]])
    counted = (sa != 0) & (sb != 0)
    return float((sa[counted] * sb[counted]).mean()) if counted.any() else NAN


# --- Verfahren auf Partien --------------------------------------------------------------------------------------------------------------------

def run_method(m: D.Matches, method: str, param: float, phi0: float = C.PHI0_DEFAULT, k0=None) -> D.Run:
    """`param`: K in Punkten (Elo) bzw. c oder τ in θ-Einheiten (Glicko, TrueSkill); `phi0` in θ-Einheiten; `k0` nur für Elo (Anlauf-K)."""
    if method == "Elo":
        return E.run_elo(m, param, k0)
    if method == "Glicko":
        return G.run_glicko(m, param, phi0)
    if method == "TrueSkill":
        return T.run_trueskill(m, param, phi0)
    raise ValueError(f"unbekanntes Verfahren: {method}")


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    matches: D.Matches
    runs: dict                 # Verfahren -> Run
    error: dict                # Verfahren -> Verfolgungsfehler (θ)
    error_regular: dict        # nur bei ungleicher Aktivität: Fehler der oft spielenden Hälfte
    error_rare: dict           # … der selten spielenden Hälfte
    cover: dict                # Verfahren -> Abdeckung (NaN bei Elo)
    final_err: dict            # Verfahren -> RMSE der Endwertung gegen die wahre Endstärke
    tau: dict                  # Verfahren -> Kendall τ der Endwertung gegen die wahre Endstärke
    player0: dict              # Verfahren -> (Fehler nach 1, 5, 20 Partien nach Eintritt bzw. Rückkehr) bei Neuling und Pause, sonst leer
    games: np.ndarray          # Partien je Spieler


@functools.lru_cache(maxsize=2048)
def analyse(s: Settings) -> Analysis:
    m = make_matches(s)
    runs = {"Elo": run_method(m, "Elo", s.k), "Glicko": run_method(m, "Glicko", s.c_theta, s.phi0_theta),
            "TrueSkill": run_method(m, "TrueSkill", s.c_theta, s.phi0_theta)}
    regular = tuple(q for q in range(m.n) if q not in m.rare)
    after = m.join_round if m.join_round else (m.pause[1] if m.pause[1] else 0)
    end_truth = m.truth[-1]
    ok = _present(m, m.rounds - 1)
    games = np.zeros(m.n, dtype=int)
    np.add.at(games, m.i, 1)
    np.add.at(games, m.j, 1)
    out = Analysis(s, m, runs, {}, {}, {}, {}, {}, {}, {}, games)
    for name, run in runs.items():
        out.error[name] = tracking_error(run, m)
        if m.rare:
            out.error_regular[name] = tracking_error(run, m, subset=regular)
            out.error_rare[name] = tracking_error(run, m, subset=m.rare)
        out.cover[name] = coverage(run, m)
        r, tr = run.final[ok], end_truth[ok]
        out.final_err[name] = float(np.sqrt(np.mean(((r - r.mean()) - (tr - tr.mean())) ** 2)))
        out.tau[name] = kendall_tau(r, tr)
        if after:
            out.player0[name] = tuple(error_after_games(run, m, g, after) for g in C.PAUSE_GAMES)
    return out


# --- Feste Instanzen und Läufe der Experimente ----------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=4096)
def _matches(kind: str, a: float, seed: int) -> D.Matches:
    """kind: "equal" (a = Drift), "active" (a = Drift, die Hälfte spielt mit P_RARE_EXP), "new" (a = Abstand, Drift 0), "pause" (a = Drift)."""
    n, d = C.EXP_N, C.EXP_DELTA
    if kind == "equal":
        return D.simulate_matches(n, d, C.EXP_ROUNDS, a, 1.0, 0, 0.0, 0, 0, seed)
    if kind == "active":
        return D.simulate_matches(n, d, C.EXP_ROUNDS, a, C.P_RARE_EXP, 0, 0.0, 0, 0, seed)
    if kind == "new":
        return D.simulate_matches(n, d, C.NEW_ROUNDS, 0.0, 1.0, C.NEW_JOIN, a, 0, 0, seed)
    if kind == "pause":
        return D.simulate_matches(n, d, C.PAUSE_ROUNDS, a, 1.0, 0, 0.0, C.PAUSE_START, C.PAUSE_LEN, seed)
    raise ValueError(kind)


@dataclass(frozen=True)
class Score:
    total: float
    regular: float
    rare: float
    cover: float
    after: tuple               # Fehler von Spieler 0 nach 1, 5, 20 Partien nach Eintritt bzw. Rückkehr
    mean_k: float


@functools.lru_cache(maxsize=65536)
def _score(kind: str, a: float, seed: int, method: str, param: float, phi0: float, k0=None) -> Score:
    m = _matches(kind, a, seed)
    run = run_method(m, method, param, phi0, k0)
    regular = tuple(q for q in range(m.n) if q not in m.rare)
    after = m.join_round if m.join_round else (m.pause[1] if m.pause[1] else 0)
    games = C.PAUSE_GAMES
    return Score(tracking_error(run, m), tracking_error(run, m, subset=regular) if m.rare else NAN, tracking_error(run, m, subset=m.rare) if m.rare else NAN,
                 coverage(run, m), tuple(error_after_games(run, m, g, after) for g in games) if after else (NAN,) * len(games), run.mean_k)


def _grid(method: str) -> tuple:
    return C.K_GRID if method == "Elo" else C.C_GRID


def _best(kind: str, a: float, method: str, phi0: float, key=lambda sc: sc.total, seeds=C.SWEEP_SEEDS) -> tuple:
    """(bester Parameter, Mittel der Scores über die Läufe als Score-Mittel); bestes nach `key` im Mittel über die festen Läufe."""
    best = None
    for p in _grid(method):
        scores = [_score(kind, a, sd, method, p, phi0) for sd in seeds]
        val = _mean([key(sc) for sc in scores])
        if best is None or val < best[0]:
            best = (val, p, scores)
    return best[1], _mean_score(best[2])


def _mean_score(scores) -> Score:
    n_after = len(scores[0].after)
    return Score(_mean([s.total for s in scores]), _mean([s.regular for s in scores]), _mean([s.rare for s in scores]), _mean([s.cover for s in scores]),
                 tuple(_mean([s.after[i] for s in scores]) for i in range(n_after)), _mean([s.mean_k for s in scores]))


# --- Experiment 1: alle spielen gleich oft ----------------------------------------------------------------------------------------------------

def sweep_equal(sigmas=C.SIGMA_SWEEP) -> list:
    """Verfolgungsfehler je Verfahren mit je bestem Abstimmparameter (K, c, τ) bei Drift σ; Anfangsunsicherheit 350 Punkte."""
    rows = []
    for sg in sigmas:
        for method in C.METHODS:
            param, sc = _best("equal", sg, method, C.PHI0_DEFAULT)
            rows.append({"sigma": sg, "method": method, "param": param, "error": sc.total, "mean_k": sc.mean_k})
    return rows


# --- Experiment 2: Kalibrierung ---------------------------------------------------------------------------------------------------------------

def sweep_calibration(cs=C.CAL_C, sigma=0.06) -> list:
    rows = []
    for c in cs:
        for method in ("Glicko", "TrueSkill"):
            sc = _mean_score([_score("equal", sigma, sd, method, c, C.PHI0_DEFAULT) for sd in C.SWEEP_SEEDS])
            rows.append({"c": c, "method": method, "cover": sc.cover, "error": sc.total})
    return rows


# --- Experiment 3: ungleiche Aktivität und die Voreinstellung φ₀ -------------------------------------------------------------------------------

ACTIVITY_VARIANTS = (("Elo", "Elo", None), ("Glicko, 350 Punkte", "Glicko", C.PHI0_DEFAULT), ("Glicko, informiert", "Glicko", C.PHI0_INFORMED),
                     ("TrueSkill, 350 Punkte", "TrueSkill", C.PHI0_DEFAULT), ("TrueSkill, informiert", "TrueSkill", C.PHI0_INFORMED))


def sweep_activity(sigma=0.06) -> list:
    rows = []
    for label, method, phi0 in ACTIVITY_VARIANTS:
        param, sc = _best("active", sigma, method, phi0 if phi0 is not None else C.PHI0_DEFAULT)
        rows.append({"label": label, "method": method, "param": param, "total": sc.total, "regular": sc.regular, "rare": sc.rare, "cover": sc.cover})
    return rows


# --- Experiment 4: Neuling --------------------------------------------------------------------------------------------------------------------

NEWCOMER_VARIANTS = (("Elo, K = 20 fest", "Elo", 20.0, C.PHI0_DEFAULT, None), ("Elo, K = 20 mit Anlauf-K 60", "Elo", 20.0, C.PHI0_DEFAULT, 60.0),
                     ("Glicko, 350 Punkte", "Glicko", 0.02, C.PHI0_DEFAULT, None), ("Glicko, informiert", "Glicko", 0.02, C.PHI0_INFORMED, None),
                     ("TrueSkill, 350 Punkte", "TrueSkill", 0.02, C.PHI0_DEFAULT, None))


def sweep_newcomer(gaps=C.NEW_GAPS) -> list:
    """Fehler des Neulings nach NEW_GAMES eigenen Partien, ohne Drift; feste Parameter (K = 20, c = τ = 0,02), nicht nachgestimmt."""
    rows = []
    gi = C.PAUSE_GAMES.index(C.NEW_GAMES)
    for gap in gaps:
        for label, method, param, phi0, k0 in NEWCOMER_VARIANTS:
            scores = [_score("new", gap, sd, method, param, phi0, k0) for sd in C.SWEEP_SEEDS]
            rows.append({"gap": gap, "label": label, "error": _mean([s.after[gi] for s in scores])})
    return rows


# --- Experiment 5: Pause und Rückkehr ---------------------------------------------------------------------------------------------------------

PAUSE_VARIANTS = (("Elo", "Elo", C.PHI0_DEFAULT), ("Glicko, informiert", "Glicko", C.PHI0_INFORMED), ("Glicko, 350 Punkte", "Glicko", C.PHI0_DEFAULT),
                  ("TrueSkill, informiert", "TrueSkill", C.PHI0_INFORMED))


def sweep_pause(sigmas=C.PAUSE_SIGMAS) -> list:
    """Fehler von Spieler 0 nach 1 / 5 / 20 Partien nach der Rückkehr, je Verfahren mit bestem Parameter nach dem Fehler nach 5 Partien."""
    rows = []
    for sg in sigmas:
        for label, method, phi0 in PAUSE_VARIANTS:
            param, sc = _best("pause", sg, method, phi0, key=lambda s: s.after[1])
            rows.append({"sigma": sg, "label": label, "param": param, "after1": sc.after[0], "after5": sc.after[1], "after20": sc.after[2]})
    return rows


# --- Experiment 6: Glicko gegen TrueSkill -----------------------------------------------------------------------------------------------------

def compare_glicko_trueskill(sigma=0.06, c=0.06) -> dict:
    """Auf denselben Partien (alle spielen, Drift σ, c = τ): Fehlerdifferenz Glicko minus TrueSkill und Kendall τ der Endwertungen."""
    diffs, taus = [], []
    for sd in C.SWEEP_SEEDS:
        m = _matches("equal", sigma, sd)
        rg, rt = run_method(m, "Glicko", c, C.PHI0_DEFAULT), run_method(m, "TrueSkill", c, C.PHI0_DEFAULT)
        diffs.append(tracking_error(rg, m) - tracking_error(rt, m))
        taus.append(kendall_tau(rg.final, rt.final))
    return {"diff_mean": _mean(diffs), "diff_sd": float(np.std(diffs)), "tau": _mean(taus), "runs": len(diffs)}


# --- Verdikte ---------------------------------------------------------------------------------------------------------------------------------

def verdict_analysis(a: Analysis):
    s, e = a.settings, a.error
    best = min(e, key=e.get)
    names = ", ".join(f"{k} {de(e[k], 3)}" for k in C.METHODS)
    text = f"Verfolgungsfehler: {names} (θ-Einheiten, kleiner ist besser)."
    if s.scenario == C.SCENARIOS[0]:
        spread = max(e.values()) - min(e.values())
        if spread < 0.03:
            return "info", text + " **Alle drei liegen gleichauf**: Wenn alle gleich oft spielen, trägt die mitgeführte Unsicherheit nichts bei."
        return "info", text + f" Am besten liegt **{best}**."
    if s.scenario == C.SCENARIOS[1]:
        r = a.error_rare
        b = min(r, key=r.get)
        return "info", text + f" Bei den selten spielenden: {', '.join(f'{k} {de(r[k], 3)}' for k in C.METHODS)}; am besten **{b}**."
    name = "des Neulings" if s.scenario == C.SCENARIOS[2] else "des Rückkehrers"
    p = a.player0
    if not p:
        return "info", text
    vals = {k: p[k][1] for k in C.METHODS}
    b = min((k for k in vals if not math.isnan(vals[k])), key=vals.get, default=None)
    if b is None:
        return "info", text
    return "info", text + (f" Fehler {name} nach 5 Partien: {', '.join(f'{k} {de(vals[k], 3)}' for k in C.METHODS)}; am besten **{b}**.")


def verdict_equal(rows):
    def at(sg, method):
        return next(r["error"] for r in rows if r["sigma"] == sg and r["method"] == method)
    sg = C.DEFAULT_SIGMA
    gap = max(abs(at(s, "Elo") - at(s, "Glicko")) for s in {r["sigma"] for r in rows})
    return "info", (f"Spielen alle gleich oft, **gibt es keinen Gewinn:** bei σ = {de(sg)} Elo {de(at(sg, 'Elo'), 3)}, Glicko {de(at(sg, 'Glicko'), 3)}, TrueSkill "
                    f"{de(at(sg, 'TrueSkill'), 3)}; der größte Abstand zwischen Elo und Glicko über alle Driften ist {de(gap, 3)}. Alle drei sind derselbe Filter mit anderem "
                    "Weg zur Verstärkung; der Abstimmparameter folgt der Drift wie das K bei Elo.")


def verdict_calibration(rows):
    def cov(c, method="Glicko"):
        return next(r["cover"] for r in rows if r["c"] == c and r["method"] == method)
    lo, hi = min(r["c"] for r in rows), max(r["c"] for r in rows)
    mid = 0.06
    return "warning", (f"Die Abdeckung des 95-%-Intervalls hängt an c: **{pct(cov(lo))} %** bei c = {de(lo)}, **{pct(cov(mid))} %** bei c = {de(mid)} (der Drift entsprechend) "
                       f"und **{pct(cov(hi))} %** bei c = {de(hi)}. Mit zu kleinem c ist die Zahl überzeugter, als sie sein darf; mit zu großem c ist das Intervall so weit, "
                       "dass es nichts sagt (und der Fehler steigt).")


def verdict_activity(rows):
    by = {r["label"]: r for r in rows}
    el, g350, ginf = by["Elo"], by["Glicko, 350 Punkte"], by["Glicko, informiert"]
    return "info", (f"Bei ungleicher Aktivität (die Hälfte spielt mit p = {de(C.P_RARE_EXP, 1)}) liegt Elo mit K = {de(el['param'], 0)} bei {de(el['total'], 3)}; **Glicko mit der "
                    f"Voreinstellung 350 Punkte** bei {de(g350['total'], 3)} (schlechter), **mit informiertem Prior** bei {de(ginf['total'], 3)} (besser). Der Gewinn hängt an "
                    "φ₀, nicht am Verfahren.")


def verdict_newcomer(rows):
    def at(gap, label):
        return next(r["error"] for r in rows if r["gap"] == gap and r["label"] == label)
    far = max(r["gap"] for r in rows)
    return "info", (f"Ein Neuling mit Abstand {de(far, 0)} vom Feld: Fehler nach {C.NEW_GAMES} Partien bei Elo mit festem K {de(at(far, 'Elo, K = 20 fest'), 2)}, mit Anlauf-K "
                    f"{de(at(far, 'Elo, K = 20 mit Anlauf-K 60'), 2)}, bei **Glicko {de(at(far, 'Glicko, 350 Punkte'), 2)}**. Gleichstark wie das Feld (Abstand 0) ist es "
                    f"umgekehrt: Elo {de(at(0.0, 'Elo, K = 20 fest'), 2)}, Glicko {de(at(0.0, 'Glicko, 350 Punkte'), 2)}. Die Unsicherheit macht Abweichler schnell, "
                    "Unauffällige verrauscht.")


def verdict_pause(rows):
    def at(sg, label, key):
        return next(r[key] for r in rows if r["sigma"] == sg and r["label"] == label)
    big, small = max(r["sigma"] for r in rows), min(r["sigma"] for r in rows)
    gain_big = at(big, "Elo", "after20") - at(big, "Glicko, informiert", "after20")
    gain_small = at(small, "Elo", "after20") - at(small, "Glicko, informiert", "after20")
    return "info", (f"Nach {C.PAUSE_LEN} Runden Pause bei starker Drift (σ = {de(big)}) liegt der Fehler nach {C.PAUSE_GAMES[2]} Partien bei **Glicko {de(at(big, 'Glicko, informiert', 'after20'), 2)}** "
                    f"gegen Elo {de(at(big, 'Elo', 'after20'), 2)} (Vorsprung {de(gain_big, 2)}); bei schwacher Drift (σ = {de(small)}) sind es "
                    f"{de(at(small, 'Glicko, informiert', 'after20'), 2)} gegen {de(at(small, 'Elo', 'after20'), 2)} (Vorsprung {de(gain_small, 2)}). Der Gewinn der Unsicherheit nach einer "
                    "Pause ist klein und bei starker Drift am größten.")


def verdict_compare(res):
    return "info", (f"Auf denselben {res['runs']} Läufen unterscheiden sich Glicko und TrueSkill im Fehler um {de(res['diff_mean'], 3)} ± {de(res['diff_sd'], 3)}; die Endwertungen stimmen "
                    f"zu Kendall τ = {de(res['tau'], 2)} überein. Logistisch gegen Probit, ein Gewicht g(φ) gegen v- und w-Funktion: dasselbe Verfahren in zwei Gewändern.")
