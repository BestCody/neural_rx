"""Repository locations: configs, models, and datasets live beside src/.

Install the package in editable mode (pip install -e .) so these resolve to the checkout.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "neural_rx"
