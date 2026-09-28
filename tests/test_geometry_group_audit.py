import json
import pytest
from spatial_intelligence.geometry_group_audit import open_index,accept_result,summary,finish


def fixture(tmp_path):
    path=tmp_path/'train.jsonl'
    rows=[dict(id='qa1',media=['a','b'],answer='A'),dict(id='qa2',media=['a','b'],answer='B'),
          dict(id='reverse',media=['b','a']),dict(id='single',media=['a'])]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return path


def test_grouping_preserves_all_qa_and_frame_order(tmp_path):
    db=open_index(tmp_path/'index.db',fixture(tmp_path),{'version':1})
    assert summary(db)==dict(total_rows=4,total_groups=3,pending_groups=2,completed_groups=1)
    for gid, in db.execute('SELECT gid FROM groups WHERE result IS NULL').fetchall():
        accept_result(db,gid,dict(eligible=True,reason='supported',edges=5))
    result=finish(db,{'version':1})
    assert result['eligible_ids']==['qa1','qa2','reverse']
    assert result['total']==4 and result['excluded']==1
    db.close()


def test_import_old_shard_propagates_only_identical_media(tmp_path):
    path=fixture(tmp_path);old=tmp_path/'old';old.mkdir()
    (old/'rows.rank0.jsonl').write_text(json.dumps(dict(id='qa1',eligible=True,reason='supported',edges=5))+'\n')
    db=open_index(tmp_path/'index.db',path,{},old)
    assert summary(db)['pending_groups']==1
    with pytest.raises(ValueError,match='Unfinished'):finish(db,{})
    db.close()
    db=open_index(tmp_path/'index.db',path,{},old)
    assert summary(db)['pending_groups']==1
    db.close()


def test_conflicting_prior_results_fail_closed(tmp_path):
    path=fixture(tmp_path);old=tmp_path/'old';old.mkdir()
    (old/'rows.rank0.jsonl').write_text('\n'.join(json.dumps(dict(id=i,eligible=True,reason='supported',edges=e)) for i,e in [('qa1',5),('qa2',6)]))
    with pytest.raises(ValueError,match='Conflicting'):open_index(tmp_path/'index.db',path,{},old)


def test_changed_contract_rejected(tmp_path):
    path=fixture(tmp_path);db=open_index(tmp_path/'index.db',path,{'v':1});db.close()
    with pytest.raises(ValueError,match='Changed'):open_index(tmp_path/'index.db',path,{'v':2})


def test_duplicate_question_ids_rejected(tmp_path):
    import sqlite3
    path=tmp_path/'dup.jsonl';path.write_text('\n'.join(json.dumps(dict(id='same',media=[x])) for x in ['a','b']))
    with pytest.raises(sqlite3.IntegrityError):open_index(tmp_path/'index.db',path,{})
