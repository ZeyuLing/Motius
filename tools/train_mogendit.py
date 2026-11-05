"""Train MoreDiff using the dedicated Motius trainer; supports torchrun."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motius.trainers.mogendit.run import main

if __name__ == '__main__':
    main()
