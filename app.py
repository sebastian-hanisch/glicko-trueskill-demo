"""Glicko und TrueSkill – die Zahl mit Unsicherheit - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Achtes Stück der Ranking-Linie: Elo schreibt die Wertung mit festem K fort. Glicko und TrueSkill führen zu jeder Wertung eine Unsicherheit mit, und aus ihr folgt ein K je
Spieler und Zeitpunkt. Sechs Experimente messen, wann das etwas bringt (Neuling, seltene Spieler, Pause), wann nicht (alle spielen gleich oft, im engen Feld) und wie gut die
Unsicherheit kalibriert ist. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import glt_constants as C
import glt_evaluation as ev
from glt_pdf_export import build_pdf
from glt_presets import (
    apply_preset,
    bounds,
    clamp_kept,
    current_value,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    remember,
    seed_widget,
    sync_query_params,
)
from glt_scenario import Settings
from glt_visualization import (
    build_activity,
    build_calibration,
    build_equal,
    build_newcomer,
    build_pause,
    build_trajectory,
    build_uncertainty,
)

st.set_page_config(page_title="Glicko und TrueSkill – Sebastian Hanisch", layout="wide")

SHOW = {"success": st.success, "info": st.info, "warning": st.warning}


def verdict(level, text):
    SHOW[level](text)


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _equal():
    return ev.sweep_equal()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _calibration():
    return ev.sweep_calibration()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _activity():
    return ev.sweep_activity()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _newcomer():
    return ev.sweep_newcomer()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _pause():
    return ev.sweep_pause()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _compare():
    return ev.compare_glicko_trueskill()


st.title("📊 Glicko und TrueSkill: die Zahl mit Unsicherheit")
st.markdown(
    """
Elo schreibt eine Wertung mit **festem K** fort: ein Spieler mit 1000 Partien bewegt sich genauso schnell wie ein Neuling. Diese Demo ist das **achte Stück der
Ranking-Linie** und fragt, was sich ändert, wenn man zu jeder Wertung eine **Unsicherheit mitführt**, wie **Glicko** und **TrueSkill** es tun: Das K wird dann je Spieler und
Zeitpunkt gewählt. Gemessen wird, wann das etwas bringt (ein **Neuling**, **seltene Spieler**, eine **Pause**), wann nicht (alle spielen gleich oft, im engen Feld), und dass die
Unsicherheit nur so **gut kalibriert** ist wie der Zuwachs c, den man annimmt.

Weiter unten: sechs **🔬 Experimente** mit Verdikt, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So rechnen Glicko und TrueSkill", expanded=True):
    st.markdown(
        """
1. **Wertung und Unsicherheit:** Jeder Spieler hat eine Wertung µ und eine Unsicherheit (bei Glicko die „Ratings Deviation“ RD, bei TrueSkill σ). Ein Unbewerteter beginnt
   mit großer Unsicherheit (üblich: 350 Punkte).
2. **Nach der Partie:** Der Gewinn an Wertung ist die Unsicherheit mal der Überraschung (Ergebnis minus Erwartung). Ein Unsicherer bewegt sich stark, ein Sicherer wenig; die
   Unsicherheit nimmt mit jeder Partie ab. Gegen einen unsicheren Gegner zählt die Partie weniger (Glicko: Gewicht g(RD)).
3. **Mit der Zeit:** Die Unsicherheit wächst, weil sich die Stärke ändern kann: um c je Runde (Glicko, auch ohne Partie) bzw. τ je Partie (TrueSkill). **c und τ sind die
   Annahme über die Drift**, so wie K bei Elo.
4. **Was dasselbe bleibt:** Beide sind Filter mit einer Verstärkung. Wo alle gleich oft spielen, liefern sie ähnliche Genauigkeit wie Elo mit passendem K, bei einer im Mittel höheren Verstärkung.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Spieler und Runden**")
    n = st.slider("Spieler n", *bounds("n_slider"), key="n_slider", help="Wie viele Spieler im Feld sind (jede Runde zufällig gepaart).")
    delta = st.slider("Stärkeabstand zwischen Nachbarn", *bounds("delta_slider"), step=C.DELTA_STEP, key="delta_slider",
                      help="Stärkedifferenz zweier benachbarter Spieler zu Beginn (logistische Einheiten).")
    rounds = st.slider("Runden", *bounds("rounds_slider"), key="rounds_slider", help="Jeder aktive Spieler spielt je Runde einmal (bei ungeradem Feld setzt einer aus).")
    sigma = st.slider("Drift σ je Runde", *bounds("sigma_slider"), step=C.SIGMA_STEP, key="sigma_slider",
                      help="Wie weit die wahren Stärken je Runde wandern (logistische Einheiten). 0 = konstant.")
    st.markdown("**Szenario**")
    scenario = st.selectbox("Wer spielt wann?", C.SCENARIOS, key="scenario_select")
    if scenario == C.SCENARIOS[1]:
        seed_widget("p_rare_slider")
        p_rare = st.slider("Spielwahrscheinlichkeit der seltenen Hälfte je Runde", *bounds("p_rare_slider"), step=C.P_RARE_STEP, key="p_rare_slider",
                           help="Die Hälfte der Spieler (ungerade Nummern) spielt nur mit dieser Wahrscheinlichkeit je Runde.")
        remember("p_rare_slider", p_rare)
    if scenario == C.SCENARIOS[2]:
        clamp_kept("join_slider", C.JOIN_MIN, int(rounds) - 2)
        seed_widget("join_slider")
        join = st.slider("Neuling kommt in Runde", C.JOIN_MIN, int(rounds) - 2, key="join_slider", help="Spieler 1 beginnt erst in dieser Runde; Startwertung ist das Feldmittel.")
        remember("join_slider", join)
        seed_widget("gap_slider")
        gap = st.slider("Abstand des Neulings vom Feld", *bounds("gap_slider"), step=C.GAP_STEP, key="gap_slider",
                        help="Wahre Stärke des Neulings minus Mittel des Feldes (logistische Einheiten).")
        remember("gap_slider", gap)
    if scenario == C.SCENARIOS[3]:
        clamp_kept("pause_start_slider", C.PAUSE_START_MIN, C.pause_start_max(int(rounds)))
        seed_widget("pause_start_slider")
        pause_start = st.slider("Pause beginnt in Runde", C.PAUSE_START_MIN, C.pause_start_max(int(rounds)), key="pause_start_slider",
                                help="Spieler 1 spielt in der Pause nicht, seine Stärke wandert weiter.")
        remember("pause_start_slider", pause_start)
        len_max = C.pause_len_max(int(rounds), int(pause_start))
        clamp_kept("pause_len_slider", C.PAUSE_LEN_MIN, len_max)
        seed_widget("pause_len_slider")
        pause_len = st.slider("Dauer der Pause in Runden", C.PAUSE_LEN_MIN, len_max, key="pause_len_slider")
        remember("pause_len_slider", pause_len)
    st.markdown("**Elo**")
    k = st.slider("K in Elo-Punkten", *bounds("k_slider"), key="k_slider", help="Fester Verstärkungsfaktor: Wertungsänderung je Partie ist K mal (Ergebnis minus Erwartung).")
    st.markdown("**Glicko und TrueSkill**")
    phi0 = st.slider("Anfangsunsicherheit in Punkten", *bounds("phi0_slider"), key="phi0_slider",
                     help="Unsicherheit eines Unbewerteten und Obergrenze der Unsicherheit. Die übliche Voreinstellung ist 350.")
    c = st.slider("Zuwachs der Unsicherheit je Runde (c, τ) in Punkten", *bounds("c_slider"), key="c_slider",
                  help="Annahme über die Drift: Glicko wächst um c je Runde (auch ohne Partie), TrueSkill um τ je Partie. 8 Punkte ≈ 0,046 θ-Einheiten.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Lauf würfeln", width="stretch", on_click=randomize_seed)

p_rare_v = round(float(current_value("p_rare_slider")), 2)
join_v = int(current_value("join_slider"))
gap_v = round(float(current_value("gap_slider")), 1)
pause_start_v = int(current_value("pause_start_slider"))
pause_len_v = int(current_value("pause_len_slider"))
sync_query_params({"n_slider": int(n), "delta_slider": round(float(delta), 2), "rounds_slider": int(rounds), "sigma_slider": round(float(sigma), 2),
                   "scenario_select": scenario, "p_rare_slider": p_rare_v, "join_slider": join_v, "gap_slider": gap_v, "pause_start_slider": pause_start_v,
                   "pause_len_slider": pause_len_v, "k_slider": int(k), "phi0_slider": int(phi0), "c_slider": int(c), "seed_input": int(seed)})

settings = Settings(int(n), round(float(delta), 2), int(rounds), round(float(sigma), 2), scenario, p_rare_v, join_v, gap_v, pause_start_v, pause_len_v, float(k),
                    float(phi0), float(c), int(seed))
a = _analysis(settings)
m = a.matches

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Wertungen")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Verfolgungsfehler: Elo", ev.de(a.error["Elo"], 3), delta=f"mittleres K {ev.de(a.runs['Elo'].mean_k, 0)} Punkte", delta_color="off",
          help="RMSE der zentrierten Wertungen gegen die zentrierten wahren Stärken ab Runde 30, in θ-Einheiten. Kleiner ist besser.")
m2.metric("Verfolgungsfehler: Glicko", ev.de(a.error["Glicko"], 3), delta=f"Abdeckung {ev.pct(a.cover['Glicko'])} % · mittleres K {ev.de(a.runs['Glicko'].mean_k, 0)}",
          delta_color="off", help="Abdeckung: Anteil der Fälle, in denen die wahre Stärke im 95-%-Intervall µ ± 1,96 RD liegt.")
m3.metric("Verfolgungsfehler: TrueSkill", ev.de(a.error["TrueSkill"], 3), delta=f"Abdeckung {ev.pct(a.cover['TrueSkill'])} % · mittleres K {ev.de(a.runs['TrueSkill'].mean_k, 0)}",
          delta_color="off")
m4.metric("Partien im Lauf", int(a.games.sum() // 2), delta=f"Spieler 1: {int(a.games[0])} Partien", delta_color="off")

st.session_state.setdefault("player_pick", 0)
st.session_state["player_pick"] = min(int(st.session_state["player_pick"]), settings.n - 1)
pick = st.selectbox("Spieler im Verlauf", list(range(settings.n)), format_func=lambda i: f"Spieler {i + 1}" + (" (selten)" if i in m.rare else ""), key="player_pick")
st.plotly_chart(build_trajectory(a, int(pick)), width="stretch", key="trajectory_chart")
st.caption("Wertung nach jeder Runde bei Elo, Glicko und TrueSkill (Wertungen je Runde um das Feldmittel zentriert, in Elo-Punkten um 1500), das Band zeigt µ ± 2 Unsicherheiten; "
           "gepunktet die wahre Stärke, die nur in der Simulation bekannt ist.")
st.plotly_chart(build_uncertainty(a), width="stretch", key="uncertainty_chart")
st.caption("Die Unsicherheit (Glicko RD) jedes Spielers: Sie sinkt mit jeder Partie und wächst in den Runden ohne Partie; ein Neuling beginnt bei der Anfangsunsicherheit.")

ok = m.presence[-1]
cen = {name: a.runs[name].final - a.runs[name].final[ok].mean() for name in C.METHODS}
truth = m.truth[-1] - m.truth[-1][ok].mean()
rows = []
for p in range(settings.n):
    rows.append({"Spieler": p + 1, "Partien": int(a.games[p]), "Elo": round(C.BASE + C.SCALE * float(cen["Elo"][p])), "Glicko": round(C.BASE + C.SCALE * float(cen["Glicko"][p])),
                 "RD Glicko": round(C.SCALE * float(a.runs["Glicko"].sd[-1][p])), "TrueSkill": round(C.BASE + C.SCALE * float(cen["TrueSkill"][p])),
                 "σ TrueSkill": round(C.SCALE * float(a.runs["TrueSkill"].sd[-1][p])), "wahre Stärke": round(C.BASE + C.SCALE * float(truth[p]))})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.download_button("📄 PDF der aktuellen Wertungen", data=build_pdf(a), file_name="glicko-trueskill.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was bringt die mitgeführte Unsicherheit?")
verdict(*ev.verdict_analysis(a))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wenn alle gleich oft spielen")
st.caption("8 Spieler (feste Instanz, unabhängig vom Regler), 150 Runden, alle spielen jede Runde, Anfangsunsicherheit 350 Punkte; je Verfahren der beste Abstimmparameter "
           "(K, c, τ) nach dem Verfolgungsfehler, Mittel über 60 feste Läufe.")
rows1 = _equal()
st.plotly_chart(build_equal(rows1), width="stretch", key="exp1_chart")
verdict(*ev.verdict_equal(rows1))
st.dataframe(pd.DataFrame([{"Drift σ": ev.de(r["sigma"]), "Verfahren": r["method"], "bester Parameter": ev.de(r["param"], 0) + " Punkte" if r["method"] == "Elo" else ev.de(r["param"], 3),
                            "Fehler": round(r["error"], 3), "mittleres K (Punkte)": round(r["mean_k"], 1)} for r in rows1]), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 2: Wie gut ist die Unsicherheit kalibriert?")
st.caption("Alle spielen jede Runde, Drift σ = 0,06 (feste Instanz); der Zuwachs c bzw. τ wird vorgegeben; Abdeckung des 95-%-Intervalls µ ± 1,96 Unsicherheit über 60 feste Läufe.")
rows2 = _calibration()
st.plotly_chart(build_calibration(rows2), width="stretch", key="exp2_chart")
verdict(*ev.verdict_calibration(rows2))

st.subheader("🔬 Experiment 3: Ungleiche Aktivität und die Voreinstellung 350 Punkte")
st.caption("8 Spieler, Drift σ = 0,06, die Hälfte spielt nur mit Wahrscheinlichkeit 0,2 je Runde; je Verfahren der beste Parameter; „informiert“ heißt Anfangsunsicherheit 0,7 θ-"
           "Einheiten (122 Punkte, die Streuung der Anfangsstärken), 60 feste Läufe.")
rows3 = _activity()
st.plotly_chart(build_activity(rows3), width="stretch", key="exp3_chart")
verdict(*ev.verdict_activity(rows3))
st.dataframe(pd.DataFrame([{"Variante": r["label"], "bester Parameter": ev.de(r["param"], 0) + " Punkte" if r["method"] == "Elo" else ev.de(r["param"], 3),
                            "Fehler gesamt": round(r["total"], 3), "oft spielende": round(r["regular"], 3), "selten spielende": round(r["rare"], 3),
                            "Abdeckung": "–" if r["cover"] != r["cover"] else f"{ev.pct(r['cover'])} %"} for r in rows3]), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Ein Neuling im eingespielten Feld")
st.caption("8 Spieler, konstante Stärken, der Neuling kommt nach 60 von 100 Runden mit Startwertung Feldmittel; Fehler seiner Wertung nach 20 eigenen Partien, feste Parameter "
           "(K = 20, c = τ = 0,02), 60 feste Läufe.")
rows4 = _newcomer()
st.plotly_chart(build_newcomer(rows4), width="stretch", key="exp4_chart")
verdict(*ev.verdict_newcomer(rows4))

st.subheader("🔬 Experiment 5: Pause und Rückkehr")
st.caption("8 Spieler, Spieler 1 pausiert 60 Runden (Runde 41 bis 100), danach 25 Runden; Fehler seiner Wertung nach 1, 5 und 20 Partien nach der Rückkehr, je Verfahren der "
           "beste Parameter nach dem Fehler nach 5 Partien, 60 feste Läufe.")
rows5 = _pause()
sig_choice = st.radio("Drift σ", C.PAUSE_SIGMAS, format_func=ev.de, horizontal=True, key="pause_sigma_choice")
st.plotly_chart(build_pause(rows5, sig_choice), width="stretch", key="exp5_chart")
verdict(*ev.verdict_pause(rows5))

st.subheader("🔬 Experiment 6: Glicko gegen TrueSkill")
st.caption("Alle spielen jede Runde, Drift σ = 0,06, c = τ = 0,06, Anfangsunsicherheit 350 Punkte; dieselben 60 festen Läufe.")
res6 = _compare()
verdict(*ev.verdict_compare(res6))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Drift ist bekannt** | Die Abdeckung des Intervalls folgt dem Zuwachs c: zu klein heißt zu überzeugt, zu groß heißt wertlos (Experiment 2). | WHR und TrueSkill Through Time |
| **Die Anfangsunsicherheit passt** | Die übliche Voreinstellung 350 Punkte ist für ein enges Feld mit seltenen Spielern zu groß und kostet Genauigkeit; in einem weiten Feld oder beim Neuling ist sie vorteilhaft (Experiment 3). | – |
| **Die Information ist ungleich verteilt** | Im engen Feld (Abstand 0,3) ist das K je Spieler kein Gewinn gegenüber einem passenden festen K, wenn alle gleich oft spielen (Experiment 1); in einem weiten Feld (Abstand 1,0) schlägt die Unsicherheit auch dann ein festes K. | – |
| **Nur die Vergangenheit zählt** | Spätere Partien verbessern frühere Schätzungen nicht; ein Filter sieht keine Zukunft. | WHR und TrueSkill Through Time |
| **Einzelne spielen gegeneinander** | Teams, Mehrpersonenspiele und partielles Spiel brauchen eine eigene Zerlegung. | Von der Mannschaft zum Einzelnen |
| **Ein Spiel ist Sieg oder Niederlage** | Remis, Tore und Heimvorteil haben ein eigenes Modell; TrueSkill kennt eine Remis-Marge, die hier nicht vorkommt. | Tore, Remis, Heimvorteil |
| **Glicko-1 genügt** | Glicko-2 führt zusätzlich eine Volatilität der Stärke mit; hier fest. | – |
| **Alles ist simuliert** | Stärken, Irrfahrt, Paarung und Aktivität sind Konstruktionen des Generators; die Abdeckung ist nur hier messbar. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Skala.** Wertungen $\mu$ in logistischen Einheiten; $1$ Einheit $= 400/\ln 10 \approx 173{,}7$ Elo-Punkte; die Erwartung ist $E = \sigma(\mu_i - \mu_j)$.

**Elo.** $\mu_i \leftarrow \mu_i + k\,(s - E)$, $\mu_j \leftarrow \mu_j - k\,(s - E)$ mit $k = K/173{,}7$.

**Glicko** (Glickman 1999, feste Volatilität; die Bezeichnung RD und die Deckelung bei $\varphi_0$ nach Glickmans Beschreibung auf glicko.net, nicht aus dem Papier). Unsicherheit $\varphi$ (in Punkten: RD). Zu Beginn jeder Periode $\varphi \leftarrow \min(\sqrt{\varphi^2 + c^2},\ \varphi_0)$.
Mit $g(\varphi_j) = 1/\sqrt{1 + 3\varphi_j^2/\pi^2}$, $E_j = \sigma\bigl(g(\varphi_j)(\mu - \mu_j)\bigr)$ und $v^{-1} = \sum_j g(\varphi_j)^2 E_j (1 - E_j)$:
$\varphi' = \bigl(\varphi^{-2} + v^{-1}\bigr)^{-1/2}$, $\mu' = \mu + \varphi'^2 \sum_j g(\varphi_j)(s_j - E_j)$.

**TrueSkill** (Herbrich, Minka, Graepel 2006), 1 gegen 1. Vor der Partie $\sigma^2 \leftarrow \sigma^2 + \tau^2$. Mit $c^2 = 2\beta^2 + \sigma_w^2 + \sigma_l^2$,
$t = (\mu_w - \mu_l)/c$, Remis-Marge $\varepsilon$ (hier 0), $v = \phi(t - \varepsilon/c)/\Phi(t - \varepsilon/c)$ und $w = v\,(v + t - \varepsilon/c)$:
$\mu_w \leftarrow \mu_w + \sigma_w^2 v/c$, $\mu_l \leftarrow \mu_l - \sigma_l^2 v/c$, $\sigma^2 \leftarrow \sigma^2\,(1 - \sigma^2 w/c^2)$. In logistischen Einheiten ist
$\beta = 1{,}702/\sqrt 2$, damit $\Phi(d/(\sqrt 2\,\beta)) \approx \sigma(d)$.

**Abdeckung.** Anteil der (Runde, Spieler)-Paare mit $|(\mu_i - \bar\mu) - (s_i - \bar s)| \le 1{,}96\,\varphi_i$, über die Spieler im Feld zentriert.

**Verfolgungsfehler.** $\sqrt{\tfrac1n\sum_i\bigl((\mu_i - \bar\mu) - (s_i - \bar s)\bigr)^2}$, gemittelt über die Runden ab 30.

**Zufall.** Irrfahrt mit Normalverteilung per Box-Muller aus der Gleichverteilten des Generators (SplitMix64); Paarung durch eine Zufallspermutation je Runde.

**Literatur.**
- Glickman, M. E. (1999): Parameter estimation in large dynamic paired comparison experiments. *Journal of the Royal Statistical Society Series C (Applied Statistics)*
  48 (3), 377–394, doi:10.1111/1467-9876.00159; Formeln, Deckelung bei 350 und Rechenbeispiel (Wertung 1500, RD 200 gegen drei Gegner: 1464 und RD 151,4) nach dem Text „The Glicko system“
  von der Seite des Autors (glicko.net); das Papier selbst ist im Volltext eingesehen (Gewicht g, Erwartung und Aktualisierung stimmen überein; Deckelung, Rechenbeispiel und „Ratings Deviation“ stehen dort nicht).
- Herbrich, R., Minka, T., Graepel, T. (2006): TrueSkill™: A Bayesian skill rating system. *Advances in Neural Information Processing Systems 19* (NIPS 2006), 569–576
  (im Volltext eingesehen: Gleichung (1), Tabelle 1 mit V und W, Remis-Marge, Voreinstellungen; die ausgeschriebene 1-gegen-1-Form folgt aus Tabelle 1 und steht bei Dangauthier et al. (2007), Abschnitt 3.1,
  und in der Dokumentation des Pakets `trueskill`).
- Elo, A. E. (1978): *The Rating of Chessplayers, Past and Present.* Arco Publishing, New York (nach Katalogeinträgen, das Buch wurde nicht eingesehen).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html)."
)
