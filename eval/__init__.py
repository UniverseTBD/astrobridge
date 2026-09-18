"""Self-contained classification-eval bench: does the trained model actually understand each
modality, measured against real external ground-truth labels — not caption fluency.

`eval/` is a real package (unlike the earlier `inference/` folder on the now-deleted
`modal-inference` branch, which deliberately had no `__init__.py` and paid for it with
`MODEL_REPO_ID`/`image`/`app` duplicated between `modal_app.py` and `compare.py`). Giving this
folder a package boundary means `eval/runners/*.py` can `from eval.backend import get_backend`
instead of re-declaring the Modal app/image/volume per script.

See eval/README.md for how to run each track.
"""
from __future__ import annotations
