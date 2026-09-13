"""Run a script using this Python's NumPy plus optional extra packages.

The active Python's installed packages retain precedence. Extra packages are
appended, never copied or modified. Useful with the app's bundled Python runtime.
Usage: python recommend/with_runtime.py EXTRA_SITE_PACKAGES SCRIPT [ARG ...]
"""
import runpy
import sys
from pathlib import Path

extra, script, *arguments = sys.argv[1:]
sys.path.append(str(Path(extra).resolve()))
sys.path.insert(0, str(Path(script).resolve().parent))
script = str(Path(script).resolve())
sys.argv = [script, *arguments]
runpy.run_path(script, run_name="__main__")
