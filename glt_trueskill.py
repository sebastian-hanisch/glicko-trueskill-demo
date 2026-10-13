"""TrueSkill (Herbrich, Minka, Graepel 2006) für 1 gegen 1, in θ-Einheiten mit β = 1,702/√2 (damit Φ(d/(√2 β)) ≈ σ(d)).

Je Partie: σ² ← σ² + τ² für beide (Dynamik), c² = 2β² + σ_w² + σ_l², t = (µ_w − µ_l)/c, ε' = ε/c mit der Remis-Marge ε = Φ⁻¹((1 + p_d)/2)·√2·β:
  v = φ(t − ε')/Φ(t − ε'),  w = v (v + t − ε'),
  µ_w += σ_w² v / c,  µ_l −= σ_l² v / c,  σ² ← σ² (1 − σ²/c² · w).
In den Partien dieser Demo gibt es kein Remis (p_d = 0, ε = 0); der Wert p_d = 0,10 ist die Voreinstellung der üblichen Pakete."""

import math
from statistics import NormalDist

import numpy as np

import glt_constants as C
from glt_drift import Matches, Run
from glt_elo import expected_score

SQ2PI = math.sqrt(2.0 * math.pi)


def pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQ2PI


def cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def draw_margin(draw_probability: float, beta: float = C.BETA) -> float:
    if not 0.0 <= draw_probability < 1.0:
        raise ValueError("draw_probability muss in [0, 1) liegen")
    if draw_probability == 0.0:
        return 0.0
    return NormalDist().inv_cdf((draw_probability + 1.0) / 2.0) * math.sqrt(2.0) * beta


def trueskill_update(mu_w: float, sd_w: float, mu_l: float, sd_l: float, beta: float = C.BETA, eps: float = 0.0) -> tuple:
    """Neue (µ_w, σ_w, µ_l, σ_l) nach einem Sieg von w über l; die Dynamik τ ist vorher zu addieren."""
    c2 = 2.0 * beta * beta + sd_w * sd_w + sd_l * sd_l
    c = math.sqrt(c2)
    t = (mu_w - mu_l) / c - eps / c
    v = pdf(t) / max(cdf(t), 1e-300)
    w = v * (v + t)
    new_w = sd_w * sd_w * (1.0 - sd_w * sd_w / c2 * w)
    new_l = sd_l * sd_l * (1.0 - sd_l * sd_l / c2 * w)
    return (mu_w + sd_w * sd_w / c * v, math.sqrt(max(new_w, 1e-12)),
            mu_l - sd_l * sd_l / c * v, math.sqrt(max(new_l, 1e-12)))


def run_trueskill(m: Matches, tau: float, sigma0: float, draw_probability: float = 0.0, beta: float = C.BETA) -> Run:
    """`tau` und `sigma0` in θ-Einheiten. Die Dynamik τ² wird, wie in der Standardfassung, vor jeder Partie auf die Beteiligten addiert."""
    n, rounds = m.n, m.rounds
    eps = draw_margin(draw_probability, beta)
    mu = [0.0] * n
    sd = [sigma0] * n
    mu_h = np.full((rounds, n), np.nan)
    sd_h = np.full((rounds, n), np.nan)
    gi, gj, gw, gr = m.i.tolist(), m.j.tolist(), m.win.tolist(), m.rnd.tolist()
    pres = m.presence.tolist()
    g, G = 0, len(gi)
    ks, cnt = 0.0, 0
    for t in range(rounds):
        if m.join_round and t == m.join_round:
            mu[0] = sum(mu[1:]) / (n - 1)
            sd[0] = sigma0
        while g < G and gr[g] == t:
            i, j, w = gi[g], gj[g], gw[g]
            win, lose = (i, j) if w else (j, i)
            sd[win] = math.sqrt(sd[win] ** 2 + tau * tau)
            sd[lose] = math.sqrt(sd[lose] ** 2 + tau * tau)
            e = expected_score(mu[win], mu[lose])
            old = mu[win]
            mu[win], sd[win], mu[lose], sd[lose] = trueskill_update(mu[win], sd[win], mu[lose], sd[lose], beta, eps)
            if 1.0 - e > 1e-9:
                ks += 2.0 * C.SCALE * abs(mu[win] - old) / (1.0 - e)
                cnt += 2
            g += 1
        for q in range(n):
            if pres[t][q]:
                mu_h[t, q] = mu[q]
                sd_h[t, q] = sd[q]
    return Run(mu_h, sd_h, np.array(mu), ks / max(cnt, 1))
