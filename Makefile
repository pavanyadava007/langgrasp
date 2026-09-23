PY ?= .venv/bin/python
export MUJOCO_GL ?= egl

.PHONY: gate lint test smoke report video gui gui-build

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

# The web GUI. One command starts the API, the simulation worker and the built frontend on one port; reach it
# with `ssh -L 8000:localhost:8000 <host>`. GUI_PORT overrides the port when 8000 is taken.
GUI_PORT ?= 8000

gui:
	$(PY) -m langgrasp.gui --port $(GUI_PORT)

# Rebuild the frontend into langgrasp/gui/static (needs Node; the build is committed so `make gui` does not).
gui-build:
	cd gui && npm install --no-audit --no-fund && npm run build
