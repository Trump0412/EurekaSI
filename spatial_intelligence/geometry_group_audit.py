"""Single-writer, recoverable geometry-only grouping; every QA row is retained."""
import json
import sqlite3
from pathlib import Path


def open_index(path, manifest, contract, previous=None):
    db=sqlite3.connect(str(path),timeout=120)
    db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
    encoded=json.dumps(contract,sort_keys=True)
    existing=db.execute("SELECT value FROM meta WHERE key='contract'").fetchone()
    if existing and existing[0]!=encoded:
        db.close();raise ValueError('Changed audit contract')
    db.execute('CREATE TABLE IF NOT EXISTS groups (gid INTEGER PRIMARY KEY, media TEXT UNIQUE, frames INTEGER, result TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS rows (id TEXT PRIMARY KEY, gid INTEGER NOT NULL)')
    db.execute('CREATE INDEX IF NOT EXISTS groups_result ON groups(result)')
    if not db.execute("SELECT 1 FROM meta WHERE key='indexed'").fetchone():
        # These tables are private derived indices, not the source manifest.
        # Transaction rollback permits a clean restart after interrupted indexing.
        db.execute('DELETE FROM rows');db.execute('DELETE FROM groups')
        db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('contract',encoded))
        with Path(manifest).open() as src:
            for line in src:
                row=json.loads(line);media=row['media']
                if not 1<=len(media)<=32:raise ValueError('Invalid frame count')
                key=json.dumps(media,separators=(',',':'))
                result=json.dumps(dict(eligible=False,reason='single_frame',edges=0)) if len(media)==1 else None
                db.execute('INSERT OR IGNORE INTO groups(media,frames,result) VALUES (?,?,?)',(key,len(media),result))
                gid=db.execute('SELECT gid FROM groups WHERE media=?',(key,)).fetchone()[0]
                db.execute('INSERT INTO rows VALUES (?,?)',(row['id'],gid))
        db.execute('INSERT INTO meta VALUES (?,?)',('indexed','true'));db.commit()
    if previous and not db.execute("SELECT 1 FROM meta WHERE key='imported'").fetchone():
        for path in sorted(Path(previous).glob('rows.rank*.jsonl')):
            with path.open() as src:
                for line in src:
                    row=json.loads(line)
                    match=db.execute('SELECT gid FROM rows WHERE id=?',(row['id'],)).fetchone()
                    if match is None:raise ValueError('Prior audit ID absent from current manifest')
                    result={k:row[k] for k in ['eligible','reason','edges']}
                    accept_result(db,match[0],result)
        db.execute('INSERT INTO meta VALUES (?,?)',('imported','true'));db.commit()
    return db


def accept_result(db,gid,result):
    if set(result)!={'eligible','reason','edges'} or not isinstance(result['eligible'],bool):
        raise ValueError('Invalid audit result')
    old=db.execute('SELECT result FROM groups WHERE gid=?',(gid,)).fetchone()
    if old is None:raise ValueError('Unknown media group')
    if old[0] is not None and json.loads(old[0])!=result:raise ValueError('Conflicting geometry-only audit results')
    db.execute('UPDATE groups SET result=? WHERE gid=?',(json.dumps(result,sort_keys=True),gid))


def summary(db):
    total=db.execute('SELECT count(*) FROM rows').fetchone()[0]
    pending=db.execute('SELECT count(*) FROM groups WHERE result IS NULL').fetchone()[0]
    groups=db.execute('SELECT count(*) FROM groups').fetchone()[0]
    return dict(total_rows=total,total_groups=groups,pending_groups=pending,completed_groups=groups-pending)


def finish(db,contract):
    counts=summary(db)
    if counts['pending_groups']:raise ValueError('Unfinished geometry groups')
    ids=[rid for rid,result in db.execute('SELECT rows.id,groups.result FROM rows JOIN groups USING(gid) ORDER BY rows.rowid') if json.loads(result)['eligible']]
    if not ids:raise ValueError('No TIP supported samples')
    return dict(status='complete',contract=contract,eligible_ids=ids,total=counts['total_rows'],
                eligible=len(ids),excluded=counts['total_rows']-len(ids),instruction_rows_unchanged=True,
                execution='unique_ordered_media_dynamic_queue_v1',group_counts=counts)
