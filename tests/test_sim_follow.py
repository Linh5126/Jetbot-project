"""Closed-loop regression: the real follower code in scripts/sim_follow.py's simulated world."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sim = pytest.importorskip("sim_follow")

CFG = yaml.safe_load(open("configs/jetbot.yaml", encoding="utf-8"))


@pytest.mark.parametrize("name", ["stand", "crossing"])
def test_guarded_mode_follows_the_raised_hand_person_safely(name):
    stats = sim.run(name, "guarded", sim.World(), 0, CFG)
    assert sim.verdict(name, stats) == []
    assert stats["follow_t"] < 12.0  # hands at 1-2.5 s, 8 s enrolment


def test_nearest_mode_approaches_and_stops_without_any_selection():
    stats = sim.run("stand", "nearest", sim.World(), 0, CFG)
    assert sim.verdict("stand", stats) == [] and stats["follow_t"] < 2.0
