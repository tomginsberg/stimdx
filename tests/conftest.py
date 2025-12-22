"""Pytest configuration for stimdx tests."""

import sys
from pathlib import Path

# Add src to path for testing without installation
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
