import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('selection',Path(__file__).resolve().parents[1]/'scripts/run-rft-reward-ablation.py')
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


class SelectionTests(unittest.TestCase):
    def candidate(self, root, name, score, test_score, ids=('a','b')):
        folder=root/name/'runs/mixed/evaluate';folder.mkdir(parents=True)
        manifest=root/(name+'.jsonl')
        manifest.write_text(''.join(json.dumps({'id':i})+'\n' for i in ids))
        value={'status':'complete','paired_ids_verified':True,'reload_verified':True,
               'initial_checkpoint':'same-sft','eval_manifests':{'validation_4drl':str(manifest)},
               'paired':{'validation_4drl':{'rft':{'count':2,'per_task':{'x':{'count':2,'mean_answer_reward':score}}}},
                         'dsrbench':{'rft':{'accuracy':test_score}}}}
        (folder/'completion.json').write_text(json.dumps(value))
        (folder/'rft-validation_4drl.rank0.jsonl').write_text(''.join(json.dumps({'id':i,'tokens':[1,2]})+'\n' for i in ids))
        return dict(name=name,root=str(root/name),arm='mixed')

    def test_ignores_test_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            a=self.candidate(root,'a',.7,.1)
            b=self.candidate(root,'b',.6,1.)
            result=MODULE.select([a,b])
            self.assertEqual(result['winner'],'a')
            self.assertFalse(result['test_used'])

    def test_mismatched_validation_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            a=self.candidate(root,'a',.7,.1)
            b=self.candidate(root,'b',.6,1.,ids=('a','c'))
            with self.assertRaises(ValueError):MODULE.select([a,b])


if __name__=='__main__':unittest.main()
