"""Persistent, fail-closed disk budget for immutable graph cache writers.

One initial *.pt inventory is indexed. All subsequent writers must use this
helper; arbitrary external cache mutations require an explicit index rebuild.
Reservations are committed BEFORE publication, so a crashed writer consumes
budget conservatively rather than creating unaccounted files. No files deleted.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import sqlite3
import uuid

GIB = 1024 ** 3


class CacheCapacityError(RuntimeError):
    pass


@contextmanager
def writer_lock(root):
    path = Path(root) / '.capacity.lock'
    with path.open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b'0'); handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _database(root):
    db = sqlite3.connect(str(root / '.capacity.sqlite3'), timeout=60)
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS entries (name TEXT PRIMARY KEY, bytes INTEGER NOT NULL, state TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS totals (id INTEGER PRIMARY KEY CHECK(id=1), bytes INTEGER NOT NULL)')
    if db.execute('SELECT bytes FROM totals WHERE id=1').fetchone() is None:
        total = 0
        # One-time migration of preexisting graph files. Never a per-get scan.
        for path in root.iterdir():
            if path.suffix not in ('.pt','.tmp') or not path.is_file():
                continue
            size = path.stat().st_size
            db.execute('INSERT INTO entries VALUES (?, ?, ?)', (path.name, size, 'existing'))
            total += size
        db.execute('INSERT INTO totals VALUES (1, ?)', (total,))
    db.commit()
    return db


def publish_cache(root, filename, payload, *, max_bytes=512 * GIB, min_free_bytes=100 * GIB):
    """Publish once under an interprocess lock; return False for an existing key.

    Existing graph files remain readable even above a newly lowered budget.
    Failed reservations are retained, never auto-evicted; retrying the SAME key
    reuses its reservation. Orphan temporary files are reported in documentation
    and require explicit operator recovery, not recursive deletion.
    """
    if Path(filename).name != filename or not filename.endswith('.pt'):
        raise ValueError('Expected one graph-cache .pt filename')
    if not isinstance(payload, bytes):
        raise ValueError('Serialized cache payload must be bytes')
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (max_bytes, min_free_bytes)):
        raise ValueError('Capacity limits must be nonnegative integer byte counts')
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    target = root / filename
    with writer_lock(root):
        db = _database(root)
        try:
            if target.exists():
                return False
            entry = db.execute('SELECT bytes,state FROM entries WHERE name=?', (filename,)).fetchone()
            size = len(payload)
            if entry is not None and entry[0] != size:
                raise CacheCapacityError('Retry payload size differs from persistent reservation; inspect cache manually')
            if entry is not None and entry[1].startswith('reserved:'):
                abandoned = root / entry[1].split(':',1)[1]
                if abandoned.exists():
                    raise CacheCapacityError('Prior reserved temporary payload exists; explicit operator recovery required, no duplicate write')
            used = db.execute('SELECT bytes FROM totals WHERE id=1').fetchone()[0]
            added = 0 if entry else size
            free = shutil.disk_usage(root).free
            if used + added > max_bytes or free - size < min_free_bytes:
                raise CacheCapacityError(
                    f'Graph cache capacity blocked: accounted={used}, additional={added}, '
                    f'cap={max_bytes}, filesystem_free={free}, required_free_after_write={min_free_bytes}; '
                    'existing cache preserved; no graph/frame truncation or automatic eviction')
            temporary = root / f'.{filename}.{os.getpid()}.{uuid.uuid4().hex}.tmp'
            if entry is None:
                db.execute('INSERT INTO entries VALUES (?, ?, ?)', (filename, size, 'reserved:'+temporary.name))
                db.execute('UPDATE totals SET bytes=bytes+? WHERE id=1', (size,))
            else:
                db.execute('UPDATE entries SET state=? WHERE name=?', ('reserved:'+temporary.name,filename))
            db.commit()  # crash-safe upper-bound accounting before file creation
            with temporary.open('xb') as handle:
                handle.write(payload); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, target)
            db.execute('UPDATE entries SET state=? WHERE name=?', ('published', filename))
            db.commit()
            return True
        finally:
            db.close()
