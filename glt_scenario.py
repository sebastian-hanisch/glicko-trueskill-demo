"""Szenario: Reglerwerte (Settings) und die daraus erzeugten Partien."""

from dataclasses import dataclass

import glt_constants as C
import glt_drift as D


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    delta: float = C.DEFAULT_DELTA
    rounds: int = C.DEFAULT_ROUNDS
    sigma: float = C.DEFAULT_SIGMA
    scenario: str = C.DEFAULT_SCENARIO
    p_rare: float = C.DEFAULT_P_RARE          # nur im Szenario „Ungleiche Aktivität“
    join: int = C.DEFAULT_JOIN                # nur im Szenario „Neuling kommt später“
    gap: float = C.DEFAULT_GAP
    pause_start: int = C.DEFAULT_PAUSE_START  # nur im Szenario „Pause und Rückkehr“
    pause_len: int = C.DEFAULT_PAUSE_LEN
    k: float = C.DEFAULT_K                    # Elo, Punkte
    phi0: float = C.DEFAULT_PHI0              # Anfangsunsicherheit von Glicko und TrueSkill, Punkte
    c: float = C.DEFAULT_C                    # Zuwachs der Unsicherheit je Runde (Glicko c, TrueSkill τ), Punkte
    seed: int = C.DEFAULT_SEED

    @property
    def p_rare_eff(self) -> float:
        return self.p_rare if self.scenario == C.SCENARIOS[1] else 1.0

    @property
    def join_round(self) -> int:
        """Runde, in der der Neuling kommt (0 = keiner); begrenzt auf [2, Runden − 2]."""
        if self.scenario != C.SCENARIOS[2]:
            return 0
        return min(max(self.join, C.JOIN_MIN), self.rounds - 2)

    @property
    def pause(self) -> tuple:
        """(Start, Länge) der Pause von Spieler 0 oder (0, 0); die Pause liegt vollständig im Lauf und lässt mindestens fünf Runden davor und danach."""
        if self.scenario != C.SCENARIOS[3]:
            return 0, 0
        start = min(max(self.pause_start, C.PAUSE_START_MIN), C.pause_start_max(self.rounds))
        length = min(max(self.pause_len, C.PAUSE_LEN_MIN), C.pause_len_max(self.rounds, start))
        return start, length

    @property
    def phi0_theta(self) -> float:
        return self.phi0 / C.SCALE

    @property
    def c_theta(self) -> float:
        return self.c / C.SCALE


def make_matches(s: Settings) -> D.Matches:
    """Deterministisch aus den Reglerwerten: gleiche Eingabe, gleiche Partien."""
    start, length = s.pause
    return D.simulate_matches(s.n, s.delta, s.rounds, s.sigma, s.p_rare_eff, s.join_round, s.gap, start, length, s.seed)
