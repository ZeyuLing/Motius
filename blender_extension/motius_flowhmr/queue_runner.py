# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent serial queue service; survives Blender and .blend changes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

if __package__:
    from . import jobs
    from .task_queue import Queue, runner_lock, smaller_schedule
else:
    import jobs
    from task_queue import Queue, runner_lock, smaller_schedule


def process_birth(pid):
    """PID reuse protection, using kernel creation time rather than a PID alone."""
    try:
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.OpenProcess.restype=wintypes.HANDLE
            kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
            kernel.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
            kernel.CloseHandle.argtypes=[wintypes.HANDLE]
            handle=kernel.OpenProcess(0x1000,False,int(pid))
            if not handle:return None
            try:
                values=[wintypes.FILETIME() for _ in range(4)]
                if not kernel.GetProcessTimes(handle,*[ctypes.byref(v) for v in values]):return None
                return str(values[0].dwHighDateTime<<32 | values[0].dwLowDateTime)
            finally:kernel.CloseHandle(handle)
        return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
    except (OSError,ValueError,IndexError):
        return None


def alive(pid,birth=None):
    if not pid:return False
    if birth is not None and process_birth(pid)!=str(birth):return False
    if os.name == 'nt':
        # os.kill(pid, 0) terminates on Windows. Never use it as a liveness test.
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:return ctypes.get_last_error()==5
        try:
            code=wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code)) and code.value==259)
        finally:kernel.CloseHandle(handle)
    try:
        os.kill(pid,0)
        state=Path(f'/proc/{pid}/stat')
        return not (state.exists() and state.read_text().split(') ',1)[1].startswith('Z'))
    except (OSError,ValueError):return False


def cancel_recovered(pid,birth,force=False):
    if birth is None or not alive(pid,birth):return
    if os.name == 'nt':
        subprocess.run(['taskkill','/PID',str(pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    else:
        import signal
        # Each worker starts its own session; never signal a foreign group.
        try:
            if os.getpgid(pid)==pid:os.killpg(pid,signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:pass


def recover(queue):
    # Never overlap an orphan worker or kill a process identified by pid alone.
    for row in queue.list(100000):
        if row['state'] not in {'running','retrying'}:
            continue
        cancel_started=None
        while alive(row['worker_pid'],row.get('worker_birth')):
            queue.update(row['id'],message='Reconnecting to previous worker; waiting for it to exit')
            if queue.get(row['id'])['cancel']:
                cancel_started=cancel_started or time.monotonic()
                cancel_recovered(row['worker_pid'],row.get('worker_birth'),time.monotonic()-cancel_started>3)
            time.sleep(.5)
        status = queue.status({**row,'state':'running'})
        if queue.get(row['id'])['cancel']:
            queue.update(row['id'],state='cancelled',worker_pid=None,message='Cancelled after reconnect')
            continue
        completed = status.get('state') == 'completed'
        queue.update(row['id'],state='completed' if completed else 'interrupted', worker_pid=None,
                     message='Result ready · import when convenient' if completed else 'Runner interrupted. Retry to resume as a new attempt.')


def execute(queue, row):
    request = json.loads(row['request'])
    for retry in range(5):
        number = row['attempt']+retry+1
        directory = Path(row['directory'])/f'attempt-{number:03d}'
        attempt_request = dict(request,output_dir=str(directory))
        queue.update(row['id'],state='running',attempt=number,retry_count=retry,message='Starting preflight',worker_pid=None)
        queue.attempt(row['id'],number,directory,attempt_request)
        job = None
        try:
            if queue.get(row['id'])['cancel']:
                queue.update(row['id'],state='cancelled',message='Cancelled before starting')
                return
            job = jobs.Job(row['python'],attempt_request)
            queue.update(row['id'],worker_pid=job.process.pid,worker_birth=process_birth(job.process.pid))
            while job.process.poll() is None:
                if queue.get(row['id'])['cancel']:
                    job.stop()
                    if time.monotonic()-job.cancel_started > 3:
                        job.stop(force=True)
                time.sleep(.25)
            code = job.process.wait()
            status = job.read_status()
            if queue.get(row['id'])['cancel']:
                state,message = 'cancelled','Cancelled · partial files retained'
            elif code == 0 and status.get('state') == 'completed':
                state,message = 'completed','Result ready · import when convenient'
            else:
                state,message = 'failed',status.get('message','Worker exited; see job.log')
                if status.get('state') != 'failed':
                    message=f"Worker exited with code {code} during {status.get('phase','startup')}. Check job.log and available system/GPU memory."
            queue.attempt(row['id'],number,directory,attempt_request,state,message)
            # Only the explicit CUDA OOM code triggers automatic retries.
            if state == 'failed' and status.get('error_code') == 'cuda_oom' and retry < 4:
                adjusted = smaller_schedule(request)
                if adjusted is not None:
                    # Preflight may have selected a schedule smaller than submitted.
                    actual = status.get('schedule',{})
                    adjusted = smaller_schedule(dict(request,**actual))
                if adjusted is not None:
                    request = adjusted
                    queue.update(row['id'],state='retrying',worker_pid=None,message='GPU memory exhausted · retrying with smaller batches')
                    continue
            queue.update(row['id'],state=state,message=message,worker_pid=None)
            return
        except Exception as error:
            queue.attempt(row['id'],number,directory,attempt_request,'failed',str(error))
            queue.update(row['id'],state='failed',message=f'Queue worker error: {error}',worker_pid=None)
            return
        finally:
            if job is not None:
                if job.process.poll() is None:
                    job.stop(force=True);job.process.wait(timeout=10)
                job.close()


def run(database, idle_seconds=30):
    queue = Queue(database)
    with runner_lock(queue.path) as acquired:
        if not acquired:
            return
        recover(queue)
        idle = time.monotonic()
        while True:
            row = queue.claim()
            if row is None:
                if time.monotonic()-idle >= idle_seconds:
                    return
                time.sleep(.5)
                continue
            execute(queue,row)
            idle = time.monotonic()


def start(python, database=None):
    queue = Queue(database)
    log = queue.path.with_suffix('.runner.log')
    with log.open('ab',buffering=0) as stream:
        subprocess.Popen([str(python),str(Path(__file__).resolve()),'--database',str(queue.path)],
                         stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                         env=jobs.clean_environment(),shell=False,
                         **({'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS} if os.name == 'nt' else {'start_new_session':True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',required=True)
    args = parser.parse_args()
    run(args.database)
