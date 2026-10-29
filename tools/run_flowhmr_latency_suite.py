"""Run bounded serial profiles, yielding the GPU if another CUDA process appears."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def apps():
    output = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return {int(line.strip()) for line in output.splitlines() if line.strip().isdigit()}


def foreign_apps(process_group, allowed=()):
    foreign = set()
    for pid in apps()-set(allowed):
        try:
            if os.getpgid(pid) != process_group:
                foreign.add(pid)
        except ProcessLookupError:
            pass
    return foreign


def stop_own_group(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    except ProcessLookupError:
        child.wait()


def cpu_ticks(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[11])+int(fields[12])
    except FileNotFoundError:
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--body-model', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--idle-resident-pid', type=int, action='append', default=[])
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--output', default='measurements')
    parser.add_argument('--timeout', type=float, default=420)
    parser.add_argument('--variants', nargs='+', default=[
        'lightweight-prior', 'lightweight-optimized',
        'lightweight-optimized', 'lightweight-prior'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for run, variant in enumerate(args.variants):
        if apps()-set(args.idle_resident_pid):
            raise RuntimeError('GPU no longer idle; no foreign process was stopped')
        dest = out / f'{run}-{variant}'
        command = [sys.executable, str(root/'tools/profile_flowhmr_latency.py'),
            '--checkpoint', args.checkpoint,
            '--body-model', args.body_model,
            '--video', args.video, '--assets', args.assets,
            '--output', str(dest), '--frames', str(args.frames), '--variant', variant]
        env = os.environ.copy()
        env.update(PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4', PYTHONFAULTHANDLER='1',
                   PYTHONUNBUFFERED='1', UCX_HANDLE_ERRORS='none')
        with open(out/f'{run}.log', 'w') as log:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            start = time.monotonic()
            resident_ticks = {pid: cpu_ticks(pid) for pid in args.idle_resident_pid}
            last_check = start
            try:
                while child.poll() is None:
                    now = time.monotonic()
                    ticks = {pid: cpu_ticks(pid) for pid in args.idle_resident_pid}
                    resident_active = any(ticks[pid]-resident_ticks[pid] > 0.25*os.sysconf("SC_CLK_TCK")*(now-last_check) for pid in ticks)
                    resident_ticks, last_check = ticks, now
                    if foreign_apps(child.pid, args.idle_resident_pid) or resident_active or now-start > args.timeout:
                        raise RuntimeError('Stopped own benchmark: competing CUDA workload or time limit')
                    time.sleep(1)
            finally:
                # Only terminate the process group created by this benchmark.
                # Video encoding children belong to that group as well.
                if child.poll() is None:
                    stop_own_group(child)
        if child.returncode:
            raise RuntimeError(f'Benchmark {run} failed; see {out}/{run}.log')
        results.append(json.loads((dest/'profile.json').read_text()))
        (out/'summary.json').write_text(json.dumps(results, indent=2))
    print('SUITE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
