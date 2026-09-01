"""Put the project root on sys.path so tests import the modules under test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
