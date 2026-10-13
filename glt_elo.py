"""Elo (Vergleichsverfahren, wie elo-demo): Aktualisierung Partie für Partie mit festem K, optional Anlauf-K für Neulinge. Rechnet in θ-Einheiten, k = K/SCALE."""

import math

import numpy as np

import glt_constants as C
from glt_drift import Matches, Run


def expected_score(a: float, b: float) -> float:
    """E = σ(a − b) in Skalarrechnung, ohne Überlauf."""
    x = float(a) - float(b)
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def effective_k(K: float, K0, played: float, g0: float = C.G0) -> float:
    """K in Punkten: fest, oder mit Anlauf K0 für Neulinge, nach g0 Partien halbiert, strebt gegen K."""
    if K0 is None:
        return K
    return K + (K0 - K) / (1.0 + played / g0)


def run_elo(m: Matches, K: float, K0=None, g0: float = C.G0) -> Run:
    n, rounds = m.n, m.rounds
    R = [0.0] * n
    played = [0] * n
    hist = np.full((rounds, n), np.nan)
    gi, gj, gw, gr = m.i.tolist(), m.j.tolist(), m.win.tolist(), m.rnd.tolist()
    pres = m.presence.tolist()
    g, G = 0, len(gi)
    ks, cnt = 0.0, 0
    for t in range(rounds):
        if m.join_round and t == m.join_round:
            R[0] = sum(R[1:]) / (n - 1)                        # Startwertung des Neulings: Mittel des Feldes
        while g < G and gr[g] == t:
            i, j, w = gi[g], gj[g], float(gw[g])
            e = expected_score(R[i], R[j])
            ki, kj = effective_k(K, K0, played[i], g0), effective_k(K, K0, played[j], g0)
            R[i] += ki / C.SCALE * (w - e)
            R[j] -= kj / C.SCALE * (w - e)
            played[i] += 1
            played[j] += 1
            ks += ki + kj
            cnt += 2
            g += 1
        for q in range(n):
            if pres[t][q]:
                hist[t, q] = R[q]
    return Run(hist, None, np.array(R), ks / max(cnt, 1))
