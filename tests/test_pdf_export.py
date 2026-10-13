"""PDF-Export: erzeugt für jedes Szenario gültige PDF-Bytes; der Latin-1-Filter ersetzt alle Zeichen außerhalb der Kernschrift."""

import glt_constants as C
import glt_evaluation as ev
import glt_pdf_export as X
from glt_scenario import Settings


def test_pdf_is_built_for_every_scenario():
    for s in (Settings(n=6, rounds=40), Settings(n=6, rounds=40, scenario=C.SCENARIOS[1], p_rare=0.3),
              Settings(n=6, rounds=40, scenario=C.SCENARIOS[2], join=15), Settings(n=5, rounds=40, scenario=C.SCENARIOS[3], pause_start=10, pause_len=10)):
        data = X.build_pdf(ev.analyse(s))
        assert data[:5] == b"%PDF-" and len(data) > 1500


def test_latin_filter_replaces_greek_and_math_symbols():
    out = X._latin("τ ≈ 0,5; σ → φ µ θ − **fett** €")
    assert out == "Tau ca. 0,5; sigma -> phi mu theta - fett EUR"
    assert out.encode("latin-1")
