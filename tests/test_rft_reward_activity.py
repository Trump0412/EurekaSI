import unittest
from spatial_intelligence.rft_reward_activity import group_activity, merge_activity, activity_gate


def score(answer, fmt=0, words=0):
    return dict(answer=answer, structure=fmt, words=words, total=answer+.5*fmt+.05*words)


CFG = dict(version='geopsro-independent-answer-v3', structure_weight=.5, words_weight=.05)


class ActivityTests(unittest.TestCase):
    def test_bare_correct_answers_do_not_pass_shaping_gate(self):
        a = group_activity([score(0), score(1)], CFG)
        self.assertEqual(activity_gate(a, CFG)['failures'], ['format_never_rewarded', 'words_never_rewarded'])

    def test_answer_only_can_pass(self):
        cfg = dict(CFG, structure_weight=0, words_weight=0)
        self.assertTrue(activity_gate(group_activity([score(0), score(1)], cfg), cfg)['accepted'])

    def test_positive_constant_format_is_not_advantage_evidence(self):
        a = group_activity([score(0,1), score(1,1)], CFG)
        self.assertEqual(a['format_advantage_groups'], 0)
        self.assertIn('format_no_group_advantage_effect', activity_gate(a, CFG)['failures'])

    def test_each_component_changes_advantage(self):
        a = group_activity([score(1,0,0),score(1,1,0),score(1,1,1)], CFG)
        self.assertTrue(activity_gate(a, CFG)['accepted'])
        self.assertEqual(merge_activity([a,a])['responses'], 6)

    def test_decomposition_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'decomposition'):
            group_activity([dict(score(1),total=2)], CFG)

    def test_empty_rejected(self):
        self.assertFalse(activity_gate({}, CFG)['accepted'])


if __name__ == '__main__':
    unittest.main()
