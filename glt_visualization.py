"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import glt_constants as C

COLORS = {"Elo": C.COLOR_ELO, "Glicko": C.COLOR_GLICKO, "TrueSkill": C.COLOR_TRUESKILL}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def to_points(theta):
    """θ (logistische Einheiten, relativ zum Feldmittel) in Elo-Punkte um 1500."""
    return C.BASE + C.SCALE * np.asarray(theta, dtype=float)


def _centered(values, ok):
    """Je Runde um das Mittel der Spieler im Feld zentrieren (NaN für Spieler außerhalb)."""
    v = np.where(ok, values, np.nan)
    return v - np.nanmean(v, axis=1)[:, None]


def build_trajectory(a, player: int):
    """Wertung des gewählten Spielers nach jeder Runde bei allen drei Verfahren, mit Band µ ± 2σ bei Glicko und TrueSkill, und die wahre Stärke gepunktet."""
    m = a.matches
    ok = m.presence
    rounds = np.arange(1, m.rounds + 1)
    fig = go.Figure()
    truth = to_points(_centered(m.truth, ok)[:, player])
    for name in C.METHODS:
        run = a.runs[name]
        mu = _centered(run.mu, ok)[:, player]
        if run.sd is not None:
            sd = run.sd[:, player] * C.SCALE
            lo, hi = to_points(mu) - 2 * sd, to_points(mu) + 2 * sd
            fig.add_trace(go.Scatter(x=np.concatenate([rounds, rounds[::-1]]), y=np.concatenate([hi, lo[::-1]]), fill="toself", mode="lines",
                                     line=dict(width=0), fillcolor=C.COLOR_BAND if name == "Glicko" else "rgba(217,140,31,0.15)", hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=rounds, y=to_points(mu), mode="lines", name=name, line=dict(color=COLORS[name], width=2.5), connectgaps=False,
                                 hovertemplate=name + ", Runde %{x}: %{y:.0f} Punkte<extra></extra>"))
    fig.add_trace(go.Scatter(x=rounds, y=truth, mode="lines", name="wahre Stärke", line=dict(color=C.COLOR_TRUTH, width=1.5, dash="dot"),
                             hovertemplate="wahr, Runde %{x}: %{y:.0f}<extra></extra>"))
    if m.join_round:
        fig.add_vline(x=m.join_round + 1, line_dash="dash", line_color="rgba(150,150,150,0.7)", annotation_text="Neuling kommt", annotation_position="top left")
    if m.pause[1] > m.pause[0]:
        fig.add_vrect(x0=m.pause[0] + 1, x1=m.pause[1], fillcolor="rgba(150,150,150,0.15)", line_width=0, annotation_text="Pause", annotation_position="top left")
    values = [truth] + [to_points(_centered(a.runs[name].mu, ok)[:, player]) for name in C.METHODS]
    lo = min(float(np.nanmin(v)) for v in values) - 80.0                 # die Achse folgt den Wertungen, nicht der anfangs großen Unsicherheit (das Band wird abgeschnitten)
    hi = max(float(np.nanmax(v)) for v in values) + 80.0
    fig.update_xaxes(title_text="Runde")
    fig.update_yaxes(title_text="Wertung in Elo-Punkten (Band: ± 2 Unsicherheiten)", range=[lo, hi])
    return _base(fig, height=420)


def build_uncertainty(a):
    """Unsicherheit (Glicko RD) aller Spieler über die Runden, in Punkten: wer selten spielt oder neu ist, ist unsicherer."""
    m = a.matches
    sd = a.runs["Glicko"].sd * C.SCALE
    rounds = np.arange(1, m.rounds + 1)
    fig = go.Figure()
    for p in range(m.n):
        fig.add_trace(go.Scatter(x=rounds, y=sd[:, p], mode="lines", name=f"Spieler {p + 1}", line=dict(color=C.PALETTE[p % len(C.PALETTE)], width=2), connectgaps=False,
                                 hovertemplate=f"Spieler {p + 1}, Runde %{{x}}: RD %{{y:.0f}} Punkte<extra></extra>"))
    fig.update_xaxes(title_text="Runde")
    fig.update_yaxes(title_text="Unsicherheit RD in Punkten (Glicko)")
    return _base(fig, height=320)


def build_equal(rows):
    sigmas = sorted({r["sigma"] for r in rows})
    fig = go.Figure()
    for method in C.METHODS:
        sel = [next(r for r in rows if r["sigma"] == sg and r["method"] == method) for sg in sigmas]
        fig.add_trace(go.Bar(x=[f"σ = {de(sg)}" for sg in sigmas], y=[r["error"] for r in sel], name=method, marker_color=COLORS[method],
                             hovertemplate=method + ", %{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Verfolgungsfehler (θ-Einheiten)")
    return _base(fig)


def build_calibration(rows):
    fig = go.Figure()
    for method in ("Glicko", "TrueSkill"):
        sel = sorted((r for r in rows if r["method"] == method), key=lambda r: r["c"])
        fig.add_trace(go.Scatter(x=[de(r["c"]) for r in sel], y=[100 * r["cover"] for r in sel], mode="lines+markers", name=method,
                                 line=dict(color=COLORS[method], width=3), hovertemplate=method + ", c = %{x}: %{y:.0f} %<extra></extra>"))
    fig.add_hline(y=95, line_dash="dash", line_color="rgba(150,150,150,0.9)", annotation_text="95 %", annotation_position="bottom right")
    fig.update_xaxes(title_text="Zuwachs der Unsicherheit je Runde c bzw. τ (θ-Einheiten)", type="category")
    fig.update_yaxes(title_text="Abdeckung des 95-%-Intervalls in %", range=[70, 101])
    return _base(fig)


def build_activity(rows):
    labels = [r["label"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["regular"] for r in rows], name="oft spielende Hälfte", marker_color="#5f9d95", hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(x=labels, y=[r["rare"] for r in rows], name="selten spielende Hälfte", marker_color="#c0392b", hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Verfolgungsfehler (θ-Einheiten)")
    return _base(fig)


def build_newcomer(rows):
    gaps = sorted({r["gap"] for r in rows})
    labels = []
    for r in rows:
        if r["label"] not in labels:
            labels.append(r["label"])
    palette = ("#2F6B65", "#5f9d95", "#8e6bb0", "#c9b6e0", "#d98c1f")
    fig = go.Figure()
    for k, label in enumerate(labels):
        sel = [next(r for r in rows if r["gap"] == g and r["label"] == label) for g in gaps]
        fig.add_trace(go.Bar(x=[f"Abstand {de(g, 1)}" for g in gaps], y=[r["error"] for r in sel], name=label, marker_color=palette[k % len(palette)],
                             hovertemplate=label + ", %{x}: %{y:.2f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text=f"Fehler nach {C.NEW_GAMES} eigenen Partien (θ-Einheiten)")
    return _base(fig)


def build_pause(rows, sigma):
    sel = [r for r in rows if r["sigma"] == sigma]
    fig = go.Figure()
    shades = ("#9cc5bf", "#5f9d95", "#2F6B65")
    for k, (key, label) in enumerate((("after1", "nach 1 Partie"), ("after5", "nach 5 Partien"), ("after20", "nach 20 Partien"))):
        fig.add_trace(go.Bar(x=[r["label"] for r in sel], y=[r[key] for r in sel], name=label, marker_color=shades[k], hovertemplate="%{x}, " + label + ": %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Fehler nach der Rückkehr (θ-Einheiten)")
    return _base(fig)
