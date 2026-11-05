"""Windows-native filesystem and process checks; no NVIDIA GPU required."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from blender_extension.motius_flowhmr.task_queue import runner_lock


@pytest.mark.skipif(os.name != 'nt', reason='Windows byte-range locking requires Windows')
def test_windows_queue_lock_and_spaces_across_processes(tmp_path):
    path = tmp_path / 'folder with spaces' / 'jobs.sqlite3'
    script = '''from blender_extension.motius_flowhmr.task_queue import runner_lock
import sys
with runner_lock(sys.argv[1]) as acquired:
    print('acquired' if acquired else 'busy')
'''
    with runner_lock(path) as acquired:
        assert acquired
        result = subprocess.run([sys.executable, '-c', script, str(path)], text=True,
                                capture_output=True, check=True, timeout=30)
        assert result.stdout.strip() == 'busy'
    with runner_lock(path) as acquired:
        assert acquired
