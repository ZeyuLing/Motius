"""Standalone entry for motius.motion.root_motion, without optional model imports."""
from pathlib import Path
import runpy
if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'motius/motion/root_motion.py'), run_name='__main__')
