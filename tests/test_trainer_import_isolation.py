"""A simulator trainer must run without unrelated body/text model imports."""

import subprocess
import sys
from pathlib import Path
import pytest


def test_beyondmimic_import_does_not_load_other_method_trainers():
    root = Path(__file__).resolve().parents[1]
    code = '''
import sys
import motius.trainers as trainers
assert 'motius.trainers.flowhmr' not in sys.modules
assert 'motius.trainers.gvhmr' not in sys.modules
assert trainers.BeyondMimicTrainer.__module__ == 'motius.trainers.beyondmimic.trainer'
assert 'motius.trainers.flowhmr' not in sys.modules
assert 'motius.trainers.gvhmr' not in sys.modules
from motius.datasets.beyondmimic import BeyondMimicMotion
assert 'motius.datasets.motion.motionhub' not in sys.modules
assert 'motius.datasets.motion.dataset_union' not in sys.modules
from motius.registry import TRAINERS
assert TRAINERS.get('BeyondMimicTrainer') is trainers.BeyondMimicTrainer
'''
    subprocess.run(
        [sys.executable, '-c', code],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    'name,module',
    [
        ('SONICPPOTrainer', 'motius.trainers.sonic.trainer'),
        ('ProtoMotionsPPOTrainer', 'motius.trainers.protomotions.trainer'),
    ],
)
def test_robot_objectives_do_not_import_foreign_training_runtimes(name, module):
    root = Path(__file__).resolve().parents[1]
    code = f'''
import sys
import motius.trainers as trainers
assert getattr(trainers, {name!r}).__module__ == {module!r}
from motius.registry import TRAINERS
assert TRAINERS.get({name!r}) is getattr(trainers, {name!r})
assert 'trl' not in sys.modules
assert 'lightning' not in sys.modules
assert 'gear_sonic' not in sys.modules
assert 'protomotions' not in sys.modules
assert not any('.vendor.' in key for key in sys.modules)
assert 'motius.trainers.flowhmr' not in sys.modules
'''
    subprocess.run(
        [sys.executable, '-c', code],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
