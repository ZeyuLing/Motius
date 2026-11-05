"""Container adapter: mount exact inputs, preserve host queue and cancel only own job."""
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import pytest

from deploy.flowhmr import container_python as adapter


def fixture_request(tmp_path):
    project=tmp_path/'project';project.mkdir();(project/'motius').mkdir()
    out=tmp_path/'output, job';out.mkdir()
    video=tmp_path/'source video.mp4';video.write_bytes(b'video')
    prepared=tmp_path/'converted.mp4';prepared.write_bytes(b'prepared')
    body=tmp_path/'neutral.npz';body.write_bytes(b'body')
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    subject=tmp_path/'subjects.json';subject.write_text(json.dumps({'video':str(prepared)}))
    request=dict(project_root=str(project),output_dir=str(out),video=str(video),body_model=str(body),
                 subjects_file=str(subject),checkpoint=str(checkpoint),offline=True,mode='STANDARD')
    path=out/'request.json';path.write_text(json.dumps(request))
    return path,request


def test_worker_builds_read_only_exact_mounts_and_preserves_original_request(tmp_path):
    path,request=fixture_request(tmp_path)
    command,name=adapter.worker_command(['-u','/addon/worker.py','--request',str(path)],docker='fake-docker')
    assert command[:3]==['fake-docker','create','--name'] and name.startswith('motius-flowhmr-')
    assert ['--network','none'] == command[command.index('--network'):command.index('--network')+2]
    assert json.loads(path.read_text())==request
    rewritten=json.loads((path.parent/'container-request.json').read_text())
    assert rewritten['project_root']=='/opt/motius' and rewritten['video']==request['video']
    mounts=[next(csv.reader([command[i+1]])) for i,value in enumerate(command) if value=='--mount']
    assert any('target='+request['video'] in m and 'readonly' in m for m in mounts)
    assert any('target='+request['output_dir'] in m and 'readonly' not in m for m in mounts)
    assert any('target='+str(tmp_path/'converted.mp4') in m for m in mounts)
    assert command[-2:]==['--request',str(path.parent/'container-request.json')]


def fake_docker(tmp_path):
    path=tmp_path/'docker';path.write_text('''#!/usr/bin/env python3
import json,os,sys,time
with open(os.environ['FAKE_LOG'],'a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1]=='create':print('container-created')
elif sys.argv[1]=='start':
 open(os.environ['FAKE_STARTED'],'w').write('running')
 if os.environ.get('FAKE_SLOW'):time.sleep(30)
else:
 if sys.argv[1]=='inspect':print('0')
''');path.chmod(0o755);return path


def test_real_adapter_process_uses_owned_container_and_checks_exit(tmp_path):
    path,_=fixture_request(tmp_path);docker=fake_docker(tmp_path);log=tmp_path/'commands.jsonl'
    env=dict(os.environ,MOTIUS_FLOWHMR_DOCKER=str(docker),FAKE_LOG=str(log),FAKE_STARTED=str(tmp_path/'started'))
    completed=subprocess.run([sys.executable,str(Path(adapter.__file__)),'-u','worker.py','--request',str(path)],env=env,capture_output=True,text=True,timeout=15)
    assert completed.returncode==0,completed.stderr
    commands=[json.loads(line) for line in log.read_text().splitlines()]
    name=commands[0][commands[0].index('--name')+1]
    assert [line[0] for line in commands]==['create','start','inspect','rm']
    assert commands[1][-1]==commands[2][-1]==commands[3][-1]==name


@pytest.mark.skipif(os.name=='nt',reason='POSIX process-group signal test')
def test_cancel_removes_only_attempts_unique_container(tmp_path):
    path,_=fixture_request(tmp_path);docker=fake_docker(tmp_path);log=tmp_path/'commands.jsonl';started=tmp_path/'started'
    env=dict(os.environ,MOTIUS_FLOWHMR_DOCKER=str(docker),FAKE_LOG=str(log),FAKE_STARTED=str(started),FAKE_SLOW='1')
    child=subprocess.Popen([sys.executable,str(Path(adapter.__file__)),'-u','worker.py','--request',str(path)],env=env,start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+5
        while not started.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert started.exists()
        os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=5)
        commands=[json.loads(line) for line in log.read_text().splitlines()]
        name=commands[0][commands[0].index('--name')+1]
        assert commands[-1]==['rm','--force',name]
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()


def test_host_queue_runner_works_without_docker(tmp_path):
    runner=tmp_path/'queue_runner.py';marker=tmp_path/'done'
    runner.write_text('from pathlib import Path;import sys;Path(sys.argv[-1]).write_text("done")')
    env=dict(os.environ,MOTIUS_FLOWHMR_DOCKER=str(tmp_path/'missing'))
    subprocess.run([sys.executable,str(Path(adapter.__file__)),'-u',str(runner),str(marker)],env=env,check=True,timeout=5)
    assert marker.read_text()=='done'
