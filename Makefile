PY ?= .venv/bin/python
export MUJOCO_GL ?= egl

.PHONY: gate lint test smoke report video

gate: lint test

lint:
	.venv/bin/ruff check langgrasp tests scripts

test:
	$(PY) -m pytest -q

smoke:
	$(PY) scripts/smoke_pick.py 20

report:
	$(PY) -m langgrasp.eval.report

video:
	$(PY) scripts/make_video.py --approach modular --seeds 5000,6000,7001 --out media/demo.mp4
