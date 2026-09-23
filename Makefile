PY ?= .venv/bin/python
export MUJOCO_GL ?= egl

.PHONY: gate lint test smoke report video gui gui-build gui-test gui-test-if-available gui-e2e gui-audit

# The gate: ruff, the Python suite, and the frontend unit tests when Node is available. The browser suite is
# `make gui-e2e`: it starts a real worker and takes a couple of minutes.
gate: lint test gui-test-if-available

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

# Frontend unit tests (projection maths, formatting, run indexing). Needs Node.
gui-test:
	cd gui && npm run test

# The same, but a clean checkout without node_modules is not a failure: say so and carry on.
gui-test-if-available:
	@if [ -d gui/node_modules ]; then \
		$(MAKE) --no-print-directory gui-test; \
	else \
		echo "frontend tests skipped: gui/node_modules is missing (run make gui-build first)"; \
	fi

# End to end in a real browser against a real worker: nine checks, including that a command runs through all
# nine stages, that the overlays are painted, and that the e-stop stops the arm in under 100 ms.
gui-e2e:
	cd gui && npx playwright test

# Accessibility and responsive audit of the running GUI (axe-core across five views and two themes).
gui-audit:
	cd gui && GUI_URL=http://127.0.0.1:$(GUI_PORT) node e2e/audit.mjs
