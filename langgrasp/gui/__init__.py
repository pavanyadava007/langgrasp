"""Web GUI for the LangGrasp stack: a FastAPI server, a single simulation worker process and a React app.

The GUI is an observer. It adds no behaviour to the pipeline: stage events come from optional hook calls at
the boundaries that already exist in ``langgrasp/policies/modular.py`` and from the controller's existing
``record_fn`` contract. Numbers shown in the browser come from ``results/*.json`` or from the live run that
produced them, never from a constant in this package.
"""

from langgrasp.gui.trace import STAGES, PipelineHooks, RecordingHooks

__all__ = ["STAGES", "PipelineHooks", "RecordingHooks"]
