"""TrueSkill 1 gegen 1: v- und w-Funktion, Remis-Marge, Aktualisierung (bekannte Werte der Originaleinheiten), Lauf über Partienfolgen."""

import math

import pytest

import glt_constants as C
import glt_trueskill as T
from conftest import build

SCALE_ORIG = C.BETA / (25.0 / 6.0)               # θ-Einheiten je Einheit der Originalskala (β = 25/6)


def _update_orig(draw_probability):
    eps = T.draw_margin(draw_probability)
    s0 = (25.0 / 3.0) * SCALE_ORIG
    w = T.trueskill_update(0.0, s0, 0.0, s0, C.BETA, eps)
    return 25.0 + w[0] / SCALE_ORIG, w[1] / SCALE_ORIG, 25.0 + w[2] / SCALE_ORIG, w[3] / SCALE_ORIG


def test_two_new_players_with_the_default_draw_probability_match_the_published_values():
    mu_w, sd_w, mu_l, sd_l = _update_orig(0.10)
    assert (round(mu_w, 3), round(sd_w, 3)) == (29.396, 7.171)
    assert (round(mu_l, 3), round(sd_l, 3)) == (20.604, 7.171)


def test_two_new_players_without_draws():
    mu_w, sd_w, mu_l, sd_l = _update_orig(0.0)
    assert (round(mu_w, 3), round(sd_w, 3)) == (29.205, 7.194)
    assert (round(mu_l, 3), round(sd_l, 3)) == (20.795, 7.194)


def test_the_two_values_by_hand_without_draws():
    # c² = 2β² + 2σ² = 2·(25/6)² + 2·(25/3)² = 173,611; c = 13,176; t = 0; v = φ(0)/Φ(0) = 0,79788; Δµ = σ² v / c = 69,444 · 0,79788 / 13,176 = 4,2052
    c = math.sqrt(2 * (25 / 6) ** 2 + 2 * (25 / 3) ** 2)
    v = (1.0 / math.sqrt(2.0 * math.pi)) / 0.5
    assert (25 / 3) ** 2 * v / c == pytest.approx(4.2052, abs=1e-3)
    assert _update_orig(0.0)[0] - 25.0 == pytest.approx((25 / 3) ** 2 * v / c, abs=1e-9)


def test_draw_margin_values_and_validation():
    assert T.draw_margin(0.0) == 0.0
    assert T.draw_margin(0.10) == pytest.approx(0.1256613468 * math.sqrt(2.0) * C.BETA, rel=1e-6)          # Φ⁻¹(0,55) = 0,1256613
    with pytest.raises(ValueError, match="draw_probability"):
        T.draw_margin(1.0)
    with pytest.raises(ValueError, match="draw_probability"):
        T.draw_margin(-0.1)


def test_v_and_w_functions():
    assert T.pdf(0.0) == pytest.approx(1.0 / math.sqrt(2.0 * math.pi))
    assert T.cdf(0.0) == pytest.approx(0.5) and T.cdf(1.959964) == pytest.approx(0.975, abs=1e-6)


def test_an_upset_moves_the_ratings_more_than_an_expected_win():
    up = T.trueskill_update(-1.0, 1.0, 1.0, 1.0)          # Außenseiter gewinnt
    exp = T.trueskill_update(1.0, 1.0, -1.0, 1.0)         # Favorit gewinnt
    assert up[0] - (-1.0) > exp[0] - 1.0
    assert up[1] < 1.0 and exp[1] < 1.0 and up[1] <= exp[1]                       # die Unsicherheit sinkt, stärker beim Überraschungsergebnis


def test_the_update_stays_finite_for_huge_rating_gaps():
    mu_w, sd_w, mu_l, sd_l = T.trueskill_update(-50.0, 1.0, 50.0, 1.0)
    assert all(math.isfinite(x) for x in (mu_w, sd_w, mu_l, sd_l)) and sd_w > 0 and sd_l > 0


def test_a_larger_uncertainty_moves_further():
    a = T.trueskill_update(0.0, 2.0, 0.0, 1.0)
    assert a[0] > 0.0 and -a[2] < a[0]                    # der Unsicherere (Sieger) bewegt sich mehr als der Sicherere (Verlierer)


def test_run_adds_the_dynamics_per_game_to_both_players():
    m = build(2, 1, [[(0, 1, True)]])
    r = T.run_trueskill(m, tau=0.5, sigma0=1.0)
    mu_w, sd_w, mu_l, sd_l = T.trueskill_update(0.0, math.sqrt(1.0 + 0.25), 0.0, math.sqrt(1.0 + 0.25))
    assert r.mu[0, 0] == pytest.approx(mu_w) and r.mu[0, 1] == pytest.approx(mu_l)
    assert r.sd[0, 0] == pytest.approx(sd_w) and r.sd[0, 1] == pytest.approx(sd_l)


def test_run_does_not_grow_the_uncertainty_of_a_player_without_games():
    m = build(3, 5, [[(0, 1, True)] for _ in range(5)])
    r = T.run_trueskill(m, tau=0.5, sigma0=1.0)
    assert (r.sd[:, 2] == 1.0).all() and (r.mu[:, 2] == 0.0).all()


def test_a_newcomer_enters_at_the_field_mean_with_sigma_zero():
    games = [[(1, 2, True), (3, 4, False)] for _ in range(5)] + [[(0, 1, True), (2, 3, True)]]
    m = build(5, 6, games, join_round=5)
    r = T.run_trueskill(m, tau=0.05, sigma0=2.0)
    assert math.isnan(r.mu[4, 0]) and not math.isnan(r.mu[5, 0])
    assert r.sd[5, 0] > r.sd[5, 1]
