"""Paths shared by all reproduction subprocesses."""
import os
from pathlib import Path
PACKAGE = Path(os.environ['EGAR_PACKAGE_ROOT']).resolve()
RESULTS = Path(os.environ['EGAR_RESULTS']).resolve()
ASSETS = PACKAGE / 'assets'
BUILD = RESULTS / 'build'
BUILD.mkdir(parents=True, exist_ok=True)
