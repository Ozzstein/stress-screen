import numpy as np
import pytest
from stress_screen.analysis.aggregate import aggregate
from stress_screen.models import MethodResult, PackTopology
from stress_screen.topology import derive_topology


def _make_method(name, z):
    return MethodResult(method_name=name, z_score=z,
                        verdict="NORMAL", metadata={})


def _build_inputs(z_per_channel):
    """z_per_channel: dict[ch] -> list of 6 rest method z-scores + 1 li_plating z."""
    rest_results = {}
    li_results = {}
    for ch, z_list in z_per_channel.items():
        rest_results[ch] = [
            _make_method(f"M{i+1}", z_list[i]) for i in range(6)
        ]
        li_results[ch] = _make_method("li_plating", z_list[6])
    return rest_results, li_results


def test_composite_preserves_extreme_pathological_z():
    """A cell with one method at z=10 must still register strongly in the
    composite (old behaviour clipped to 5 then averaged with 6 zeros gave 0.71;
    new behaviour clips to 8 → 8/7 ≈ 1.14)."""
    n = 8
    z_map = {ch: [0.0] * 7 for ch in range(n)}
    z_map[0][0] = 10.0  # M1 catastrophic
    rest_r, li_r = _build_inputs(z_map)
    topo = derive_topology(n, 1)
    verdicts = aggregate(rest_r, li_r, topo)
    ch0_cv = next(cv for mv in verdicts for cv in mv.all_cells if cv.channel_index == 0)
    assert ch0_cv.composite_z > 1.0, (
        f"Pathological z=10 should still register strongly; got composite_z={ch0_cv.composite_z}"
    )


def test_composite_does_not_clip_healthy_negative_z_to_zero():
    """A consistently healthy cell (all z = -2) should produce composite < 0,
    not 0 like before."""
    n = 8
    z_map = {ch: [0.0] * 7 for ch in range(n)}
    z_map[0] = [-2.0] * 7
    rest_r, li_r = _build_inputs(z_map)
    topo = derive_topology(n, 1)
    verdicts = aggregate(rest_r, li_r, topo)
    ch0_cv = next(cv for mv in verdicts for cv in mv.all_cells if cv.channel_index == 0)
    assert ch0_cv.composite_z < 0.0, (
        f"All-negative-z cell should have negative composite_z; got {ch0_cv.composite_z}"
    )
