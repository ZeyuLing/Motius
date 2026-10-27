# SPDX-License-Identifier: GPL-3.0-or-later
"""Durable local queue, shared by Blender and an independent serial runner."""
import contextlib
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid

TERMINAL = {'completed', 'failed', 'cancelled', 'interrupted'}


def default_database():
    return Path(os.environ.get('MOTIUS_FLOWHMR_QUEUE', str(Path.home()/'.cache/motius/flowhmr/queue.sqlite3'))).expanduser().resolve()


class Queue:
    def __init__(self, path=None):
        self.path = Path(path or default_database()).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, created REAL NOT NULL, updated REAL NOT NULL,
                position INTEGER NOT NULL, state TEXT NOT NULL, request TEXT NOT NULL,
                python TEXT NOT NULL, directory TEXT NOT NULL, attempt INTEGER DEFAULT 0,
                cancel INTEGER DEFAULT 0, message TEXT DEFAULT '', runner_pid INTEGER,
                worker_pid INTEGER, worker_birth TEXT, retry_count INTEGER DEFAULT 0);
                CREATE INDEX IF NOT EXISTS task_state_position ON tasks(state,position);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts (
                task_id TEXT NOT NULL, number INTEGER NOT NULL, directory TEXT NOT NULL,
                state TEXT NOT NULL, settings TEXT NOT NULL, message TEXT DEFAULT '',
                PRIMARY KEY(task_id,number));''')
            if "worker_birth" not in {r[1] for r in db.execute("PRAGMA table_info(tasks)")}:
                db.execute("ALTER TABLE tasks ADD COLUMN worker_birth TEXT")

    @contextlib.contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA busy_timeout=10000')
        try:
            with db:
                yield db
        finally:
            db.close()

    def enqueue(self, python, request):
        identity = uuid.uuid4().hex
        request = dict(request)
        directory = Path(request['output_dir']).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        # Freeze the selected trajectory rather than depending on a mutable scene.
        if request.get('subjects_file'):
            source = Path(request['subjects_file']).resolve()
            data = json.loads(source.read_text())
            # Subjects store relative paths; resolve before copying the manifest.
            for key in ('video', 'preview_sequence'):
                if data.get(key) and not Path(data[key]).is_absolute():
                    data[key] = str((source.parent/data[key]).resolve())
            frozen = directory/'selected-subjects.json'
            frozen.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
            request['subjects_file'] = str(frozen)
        (directory/'submission.json').write_text(json.dumps(request, indent=2), encoding='utf-8')
        now = time.time()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            position = db.execute('SELECT COALESCE(MAX(position),0)+1 FROM tasks').fetchone()[0]
            db.execute('INSERT INTO tasks(id,created,updated,position,state,request,python,directory) VALUES (?,?,?,?,?,?,?,?)',
                       (identity, now, now, position, 'queued', json.dumps(request), str(python), str(directory)))
        return identity

    def get(self, identity):
        with self.connect() as db:
            row = db.execute('SELECT * FROM tasks WHERE id=?', (identity,)).fetchone()
            return dict(row) if row else None

    def list(self, limit=100):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM tasks ORDER BY CASE WHEN state IN ('running','retrying') THEN 0 WHEN state='queued' THEN 1 ELSE 2 END, CASE WHEN state IN ('queued','running','retrying') THEN position END, created DESC LIMIT ?", (limit,))]

    def update(self, identity, **values):
        allowed = {'state','attempt','cancel','message','runner_pid','worker_pid','worker_birth','retry_count','position'}
        if not values.keys() <= allowed:
            raise ValueError('Invalid queue fields')
        values['updated'] = time.time()
        with self.connect() as db:
            db.execute('UPDATE tasks SET '+','.join(k+'=?' for k in values)+' WHERE id=?', (*values.values(), identity))

    def claim(self):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            paused = db.execute("SELECT value FROM settings WHERE key='paused'").fetchone()
            if paused and paused[0] == '1': return None
            row = db.execute("SELECT * FROM tasks WHERE state='queued' AND cancel=0 ORDER BY position,created LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE tasks SET state='running',runner_pid=?,updated=? WHERE id=?", (os.getpid(),time.time(),row['id']))
            result = dict(row)
            result['state'] = 'running'
            return result

    def paused(self):
        with self.connect() as db:
            row=db.execute("SELECT value FROM settings WHERE key='paused'").fetchone()
            return bool(row and row[0]=='1')

    def pause(self, value):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO settings VALUES (?,?)',('paused','1' if value else '0'))

    def cancel(self, identity):
        with self.connect() as db:
            db.execute("UPDATE tasks SET cancel=1,state=CASE WHEN state='queued' THEN 'cancelled' ELSE state END,updated=? WHERE id=? AND state IN ('queued','running','retrying')", (time.time(),identity))

    def retry(self, identity):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT state FROM tasks WHERE id=?',(identity,)).fetchone()
            if not row or row['state'] not in {'failed','cancelled','interrupted'}:
                raise ValueError('Only failed, cancelled or interrupted tasks can be retried')
            position = db.execute('SELECT COALESCE(MAX(position),0)+1 FROM tasks').fetchone()[0]
            db.execute("UPDATE tasks SET state='queued',cancel=0,retry_count=0,message='',position=?,updated=? WHERE id=?",(position,time.time(),identity))

    def move(self, identity, delta):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rows = db.execute("SELECT id,position FROM tasks WHERE state='queued' ORDER BY position,created").fetchall()
            ids = [r['id'] for r in rows]
            if identity not in ids:
                return
            i = ids.index(identity);j = max(0,min(len(ids)-1,i+delta))
            ids[i],ids[j] = ids[j],ids[i]
            for rank, key in enumerate(ids):
                db.execute('UPDATE tasks SET position=? WHERE id=?',(rank,key))

    def attempt(self, identity, number, directory, settings, state='running', message=''):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO attempts VALUES (?,?,?,?,?,?)', (identity,number,str(directory),state,json.dumps(settings),message))

    def revision(self):
        with self.connect() as db:
            rows=db.execute("SELECT directory,attempt FROM tasks WHERE state='running'").fetchall()
        mtimes=[]
        for row in rows:
            try:mtimes.append((Path(row['directory'])/f"attempt-{row['attempt']:03d}"/'status.json').stat().st_mtime_ns)
            except OSError:mtimes.append(0)
        return (self.path.stat().st_mtime_ns,tuple(mtimes))

    def status(self, row):
        directory = Path(row['directory'])/f"attempt-{row['attempt']:03d}"
        try:
            value = json.loads((directory/'status.json').read_text(encoding='utf-8'))
        except (OSError,ValueError):
            value = {}
        if row['state'] in {'queued','retrying'}:
            value = {}
        if row['state'] != 'running':
            value.update(state=row['state'], message=row['message'] or row['state'].title())
        value['attempt_directory'] = str(directory)
        return value


@contextlib.contextmanager
def runner_lock(database):
    """OS lock is released on crashes; no stale-pid guessing or foreign kills."""
    path = Path(str(database)+'.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            if os.name == 'nt':
                stream.seek(0);msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def smaller_schedule(request):
    """No mode/dtype/observation changes; retry only bounded workspace scheduling."""
    if request.get('mode') not in {'STANDARD', 'LIGHTWEIGHT'}:
        return None
    result = dict(request)
    changed = False
    for key, default, floor in [('sam_frame_batch_size',8,1),('vggt_frame_batch_size',8,1),
                                ('vggt_query_chunk_size',4096,256),('vggt_token_chunk_size',4096,256),
                                ('query_chunk_size',256,32)]:
        value = int(request.get(key,default))
        new = max(floor,value//2)
        result[key] = new;changed |= new < value
    return result if changed else None
