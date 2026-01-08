import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def add_section_path(section_dir: str):
    """Add section folder to sys.path so flat imports like 'from layers import ...' work."""
    p = os.path.join(ROOT, section_dir)
    if p not in sys.path:
        sys.path.insert(0, p)
