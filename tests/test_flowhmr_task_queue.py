"""Durable scheduling, real child isolation, memory policy and safe OOM retries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from blender_extension.motius_flowhmr.task_queue import Queue, runner_lock, smaller_schedule
from blender_extension.motius_flowhmr import queue_runner, jobs
from blender_extension.motius_flowhmr.runtime_preflight import memory_policy, PreflightError


def request(tmp_path, **kwargs):
    return dict(output_dir=str(tmp_path),project_root=str(tmp_path),mode='LIGHTWEIGHT',
                video='complete-video.mp4',seed=17,max_frames=0,camera_interval=1,**kwargs)


def test_queue_persists_pause_reordering_cancel_retry_and_atomic_claim(tmp_path):
    path=tmp_path/'queue.sqlite3';queue=Queue(path)
    a,b=[queue.enqueue(sys.executable,request(tmp_path/name)) for name in ['a','b']]
    queue.pause(True)
    assert Queue(path).claim() is None
    queue.move(b,-1);queue.pause(False)
    assert queue.claim()['id']==b
    with ThreadPoolExecutor(2) as pool:
        rows=list(pool.map(lambda _:Queue(path).claim(),range(2)))
    assert [row['id'] for row in rows if row]==[a]
    queue.cancel(a);assert queue.get(a)['cancel']==1
    queue.update(a,state='failed');queue.retry(a)
    assert Queue(path).get(a)['state']=='queued'
    with pytest.raises(ValueError):queue.retry(b)
    c=queue.enqueue(sys.executable,request(tmp_path/'c'))
    queue.cancel(c);assert Queue(path).get(c)['state']=='cancelled'


def test_queue_freezes_selected_track_and_preserves_absolute_video(tmp_path):
    source=tmp_path/'detection/subjects.json';source.parent.mkdir()
    source.write_text(json.dumps({'video':'../source.mp4','preview_sequence':'frames/000.jpg','tracks':[{'id':1}]}))
    queue=Queue(tmp_path/'queue.sqlite3')
    identity=queue.enqueue(sys.executable,request(tmp_path/'out',subjects_file=str(source)))
    frozen=json.loads(queue.get(identity)['request'])['subjects_file']
    source.write_text('{}')
    data=json.loads(Path(frozen).read_text())
    assert data['tracks']==[{'id':1}]
    assert data['video']==str(tmp_path/'source.mp4')


def test_smaller_schedule_never_changes_observations_precision_or_mode(tmp_path):
    original=request(tmp_path,steps=20,dtype='bfloat16',postprocess=False,sam_frame_batch_size=8)
    reduced=smaller_schedule(original)
    assert reduced['sam_frame_batch_size']==4 and reduced['vggt_frame_batch_size']==4
    for key in original:
        if key!='sam_frame_batch_size':assert reduced[key]==original[key]
    while smaller_schedule(reduced) is not None:reduced=smaller_schedule(reduced)
    assert reduced['sam_frame_batch_size']==1 and reduced['vggt_query_chunk_size']==256
    standard=smaller_schedule(dict(original,mode='STANDARD',sam_frame_batch_size=1))
    assert standard['sam_frame_batch_size']==1 and standard['vggt_frame_batch_size']==4


def test_memory_policy_rejects_insufficient_free_and_reduces_only_workspace():
    with pytest.raises(PreflightError,match='available GPU memory'):
        memory_policy(3*2**30,24*2**30,750,'LIGHTWEIGHT',{})
    small=memory_policy(7*2**30,8*2**30,750,'LIGHTWEIGHT',{})
    large=memory_policy(20*2**30,24*2**30,750,'LIGHTWEIGHT',{})
    assert small['schedule']['vggt_frame_batch_size']==1
    assert large['schedule']['vggt_frame_batch_size']==8
    assert large['schedule']['sam_frame_batch_size']==8
    assert memory_policy(20*2**30,24*2**30,750,'LIGHTWEIGHT',{'sam_frame_batch_size':2})['schedule']['sam_frame_batch_size']==2
    assert 'max_frames' not in small['schedule'] and 'dtype' not in small['schedule']
    standard=memory_policy(20*2**30,24*2**30,750,'STANDARD',{})
    assert standard['schedule']['sam_frame_batch_size']==8
    with pytest.raises(PreflightError):memory_policy(20*2**30,24*2**30,10000,'LIGHTWEIGHT',{})


def test_runner_lock_excludes_second_owner(tmp_path):
    database=tmp_path/'queue.sqlite3'
    with runner_lock(database) as first:
        with runner_lock(database) as second:assert first and not second
    with runner_lock(database) as third:assert third


def scripted_job_factory(tmp_path, monkeypatch):
    script=tmp_path/'child.py'
    script.write_text('''import json,sys,time
from pathlib import Path
request=json.loads(Path(sys.argv[1]).read_text());p=Path(request['output_dir'])
if request.get('sleep'):time.sleep(request['sleep'])
if request.get('other_error'):
 result={'state':'failed','message':'bad checkpoint','error_code':'runtime'};code=1
elif request['sam_frame_batch_size']>2:
 result={'state':'failed','message':'out of memory','error_code':'cuda_oom','schedule':{'sam_frame_batch_size':request['sam_frame_batch_size']}};code=1
else:
 result={'state':'completed','message':'ready','motion_file':str(p/'motion.blender.npz')};code=0
(p/'status.json').write_text(json.dumps(result));sys.exit(code)
''')
    real_job=jobs.Job
    class Child(real_job):
        def __init__(self,python,request):
            self.directory=Path(request['output_dir']);self.directory.mkdir(parents=True)
            path=self.directory/'request.json';path.write_text(json.dumps(request))
            self.log_path=self.directory/'job.log';self.log=self.log_path.open('wb');self.cancel_started=None
            self.process=subprocess.Popen([python,str(script),str(path)],stdout=self.log,stderr=self.log,
                                          **({'start_new_session':True} if os.name!='nt' else {}))
    monkeypatch.setattr(queue_runner.jobs,'Job',Child)


def test_real_children_oom_retry_is_fresh_and_retains_attempt_history(tmp_path,monkeypatch):
    scripted_job_factory(tmp_path,monkeypatch)
    queue=Queue(tmp_path/'queue.sqlite3')
    identity=queue.enqueue(sys.executable,request(tmp_path/'out',sam_frame_batch_size=8,frontend_precision='float16'))
    queue_runner.run(queue.path,idle_seconds=0)
    row=queue.get(identity);assert row['state']=='completed' and row['attempt']==3
    assert queue.status(row)['motion_file'].endswith('attempt-003/motion.blender.npz')
    for number,batch in enumerate([8,4,2],1):
        attempt=json.loads((tmp_path/f'out/attempt-{number:03d}/request.json').read_text())
        assert attempt['sam_frame_batch_size']==batch
        assert attempt['frontend_precision']=='float16'
        assert attempt['max_frames']==0 and attempt['seed']==17 and attempt['camera_interval']==1
    with queue.connect() as db:
        assert [r[0] for r in db.execute('SELECT state FROM attempts ORDER BY number')]==['failed','failed','completed']


def test_non_oom_failure_is_not_retried_and_next_case_runs(tmp_path,monkeypatch):
    scripted_job_factory(tmp_path,monkeypatch);queue=Queue(tmp_path/'queue.sqlite3')
    a=queue.enqueue(sys.executable,request(tmp_path/'a',sam_frame_batch_size=1,other_error=True))
    b=queue.enqueue(sys.executable,request(tmp_path/'b',sam_frame_batch_size=1))
    queue_runner.run(queue.path,idle_seconds=0)
    assert queue.get(a)['state']=='failed' and queue.get(a)['attempt']==1
    assert queue.get(b)['state']=='completed'


def test_restart_marks_abandoned_attempt_and_retains_ready_result(tmp_path):
    queue=Queue(tmp_path/'queue.sqlite3')
    a=queue.enqueue(sys.executable,request(tmp_path/'a'));queue.claim();queue.update(a,attempt=1,worker_pid=None)
    b=queue.enqueue(sys.executable,request(tmp_path/'b'));queue.claim();queue.update(b,attempt=1,worker_pid=None)
    directory=tmp_path/'b/attempt-001';directory.mkdir();(directory/'status.json').write_text(json.dumps({'state':'completed','motion_file':'ready.npz'}))
    queue_runner.recover(queue)
    assert queue.get(a)['state']=='interrupted' and queue.get(b)['state']=='completed'


def test_cancelling_running_child_releases_it_and_next_task_proceeds(tmp_path,monkeypatch):
    import threading
    scripted_job_factory(tmp_path,monkeypatch);queue=Queue(tmp_path/'queue.sqlite3')
    a=queue.enqueue(sys.executable,request(tmp_path/'a',sam_frame_batch_size=1,sleep=20))
    b=queue.enqueue(sys.executable,request(tmp_path/'b',sam_frame_batch_size=1))
    thread=threading.Thread(target=queue_runner.run,args=(queue.path,0));thread.start()
    deadline=time.monotonic()+5
    while not queue.get(a)['worker_pid'] and time.monotonic()<deadline:time.sleep(.01)
    pid=queue.get(a)['worker_pid'];assert pid
    queue.cancel(a);thread.join(timeout=10)
    assert not thread.is_alive()
    assert not queue_runner.alive(pid)
    assert queue.get(a)['state']=='cancelled' and queue.get(b)['state']=='completed'


def test_recovery_waits_for_actual_orphan_and_guards_pid_reuse(tmp_path):
    queue=Queue(tmp_path/'queue.sqlite3');identity=queue.enqueue(sys.executable,request(tmp_path/'out'))
    queue.claim();directory=tmp_path/'out/attempt-001';directory.mkdir()
    script="import time,json;from pathlib import Path;time.sleep(.3);Path("+repr(str(directory/'status.json'))+").write_text(json.dumps({'state':'completed'}))"
    child=subprocess.Popen([sys.executable,'-c',script],**({'start_new_session':True} if os.name!='nt' else {}))
    try:
        birth=queue_runner.process_birth(child.pid)
        assert queue_runner.alive(child.pid)
        if birth is not None:assert not queue_runner.alive(child.pid,'wrong-birth')
        queue.update(identity,attempt=1,worker_pid=child.pid,worker_birth=birth)
        started=time.monotonic();queue_runner.recover(queue)
        assert time.monotonic()-started>=.2
        assert queue.get(identity)['state']=='completed'
    finally:
        child.wait(timeout=5)
