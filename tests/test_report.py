from langgrasp.eval import report


def test_report_builds_without_placeholders(tmp_path, monkeypatch):
    txt = report.build()
    assert txt.startswith("# Results")
    assert "\u2014" not in txt and "\u2013" not in txt
    # a missing file must render as "not run", never as a number
    assert "not run" in txt or "n=" in txt


def test_ci_formatting():
    assert report._ci(None) == "not run"
    assert report._ci({"n": 0}) == "not run"
    assert report._ci({"k": 5, "n": 10, "p": 0.5, "lo": 0.2366, "hi": 0.7634}) == "50.0% [24-76] (n=10)"
