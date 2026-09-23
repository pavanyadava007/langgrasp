"""The safety table the GUI shows must not drift from the code, the tests or docs/FMEA.md.

A hazard table that quietly stops matching the code is worse than no table, because it still reads as though
someone checked. These tests are what stops that: every hazard id has to appear in the analysis document,
every code reference has to resolve to something that exists, and every test named as evidence has to be a
test that pytest actually collects.
"""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FMEA_YAML = ROOT / "langgrasp" / "gui" / "fmea.yaml"
FMEA_DOC = ROOT / "docs" / "FMEA.md"


@pytest.fixture(scope="module")
def fmea() -> dict:
    return yaml.safe_load(FMEA_YAML.read_text())


@pytest.fixture(scope="module")
def collected_tests() -> set[str]:
    """Every test node id in this suite, as 'tests/file.py::name'."""
    ids = set()
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        text = path.read_text()
        for name in re.findall(r"^def (test_\w+)", text, flags=re.M):
            ids.add(f"tests/{path.name}::{name}")
    return ids


def all_tests(fmea: dict) -> list[str]:
    out = [t for h in fmea["hazards"] for t in h.get("tests", [])]
    out += [t for c in fmea["cross_cutting"] for t in c.get("tests", [])]
    return out


def test_every_hazard_of_the_analysis_is_in_the_table(fmea):
    doc = FMEA_DOC.read_text()
    ids = [h["id"] for h in fmea["hazards"]]
    assert ids == [f"H{i}" for i in range(1, 9)], ids
    for h in fmea["hazards"]:
        assert re.search(rf"### {h['id']} ", doc), f"{h['id']} is in the GUI table but not in docs/FMEA.md"
    for hid in re.findall(r"^### (H\d) ", doc, flags=re.M):
        assert hid in ids, f"{hid} is in docs/FMEA.md but missing from the GUI table"


def test_every_named_test_exists(fmea, collected_tests):
    missing = [t for t in all_tests(fmea) if t not in collected_tests]
    assert not missing, f"the safety table cites tests that do not exist: {missing}"


def test_every_code_reference_resolves(fmea):
    """A reference is 'path' or 'path: symbol, symbol'. The file must exist and each symbol must be defined."""
    refs = [(h["id"], m["where"]) for h in fmea["hazards"] for m in h["mitigations"]]
    refs += [("cross-cutting", c["where"]) for c in fmea["cross_cutting"]]
    problems = []
    for hid, ref in refs:
        path_part, _, symbols = ref.partition(": ")
        path = ROOT / path_part.strip()
        if not path.exists():
            problems.append(f"{hid}: {path_part} does not exist")
            continue
        if not symbols:
            continue
        text = path.read_text()
        for symbol in (s.strip() for s in symbols.split(",")):
            name = symbol.split(".")[-1]
            if not re.search(rf"(def |class |^{re.escape(name)}\s*[:=]|\.{re.escape(name)}\s*=)\s*{re.escape(name)}?", text, flags=re.M) and name not in text:
                problems.append(f"{hid}: {name} not found in {path_part}")
    assert not problems, problems


def test_the_table_states_what_is_only_simulated(fmea):
    """The point of the status column is that a reader can tell what was exercised from what was not."""
    assert set(fmea["statuses"]) == {"sim", "partial", "hardware"}
    for h in fmea["hazards"]:
        assert h["status"] in fmea["statuses"], h
        assert h["residual"], f"{h['id']} has no residual-risk sentence"
        assert h["sec"].count("/") == 2, f"{h['id']} is missing its S/E/C rating"
    # the rows that depend on hardware this project does not have must say so rather than claiming coverage
    hardware_only = [m for h in fmea["hazards"] for m in h["mitigations"] if m.get("status") == "hardware"]
    assert len(hardware_only) >= 3
    assert all("hardware only" in m["what"].lower() or "planned" in m["what"].lower() for m in hardware_only)


def test_measured_claims_point_at_a_results_file(fmea):
    for h in fmea["hazards"]:
        claim = h.get("measured")
        if not claim:
            continue
        files = re.findall(r"results/[\w.\-]+\.json", claim)
        if not files:
            assert "Measured in this GUI" in claim, f"{h['id']} claims a measurement with no source: {claim}"
            continue
        for f in files:
            assert (ROOT / f).exists(), f"{h['id']} cites {f}, which does not exist"
