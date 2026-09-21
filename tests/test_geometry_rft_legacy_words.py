import unittest

from spatial_intelligence.geometry_rft_reward import RewardConfig, score_response
from spatial_intelligence.geometry_rft_legacy_words import (
    OBSERVATION_WORDS, TRANSITION_WORDS, DERIVATION_WORDS,
    SPATIAL_RELATION_WORDS, EVIDENCE_GEOMETRY_WORDS, UNSUPPORTED_RISK_WORDS,
    _unique_phrase_hits,
)

TEXT = '''<think>
Spatial Observation: The visible object is on the left side of the image; its depth and position provide visual evidence.
Spatial Transition: The camera moves across frames while the same object remains visible and gets closer.
Answer Derivation: Therefore, based on the observation and the transition, the relative distance supports the selected option.
</think><answer>A</answer>'''


class LegacyLexicalTests(unittest.TestCase):
    def config(self, **kwargs):
        return RewardConfig(version='geopsro-lexicon-strict-v2', **kwargs)

    def test_source_inventory(self):
        self.assertEqual([len(x) for x in (OBSERVATION_WORDS, TRANSITION_WORDS,
            DERIVATION_WORDS, SPATIAL_RELATION_WORDS, EVIDENCE_GEOMETRY_WORDS,
            UNSUPPORTED_RISK_WORDS)], [163, 154, 88, 101, 82, 21])

    def test_phrases_and_correctness_gate(self):
        self.assertIn('across frames', _unique_phrase_hits('ACROSS frames', TRANSITION_WORDS))
        self.assertNotIn('across frames', _unique_phrase_hits('across frameshift', TRANSITION_WORDS))
        good = score_response(TEXT, 'A', config=self.config())
        bad = score_response(TEXT, 'B', config=self.config())
        self.assertGreater(good['words'], 0)
        self.assertEqual(bad['words'], 0)
        self.assertEqual(good['parsed_answer'], bad['parsed_answer'])
        self.assertEqual(bad['total'], .5)

    def test_ablations_share_answer(self):
        full = score_response(TEXT, 'A', config=self.config())
        fmt = score_response(TEXT, 'A', config=self.config(words_weight=0))
        ans = score_response(TEXT, 'A', config=self.config(words_weight=0, structure_weight=0))
        self.assertEqual(ans['total'], 1)
        self.assertEqual(fmt['total'], 1.5)
        self.assertAlmostEqual(full['total'] - fmt['total'], .05 * full['words'])

    def test_stuffing_and_truncation(self):
        stuffing = ', '.join(sorted(OBSERVATION_WORDS))
        text = TEXT.replace('The visible object is on the left side of the image; its depth and position provide visual evidence.', stuffing)
        self.assertEqual(score_response(text, 'A', config=self.config())['words'], 0)
        self.assertEqual(score_response(TEXT, 'A', config=self.config(), truncated=True)['total'], 0)
        self.assertEqual(score_response(TEXT+'<answer>B</answer>', 'A', config=self.config())['total'], 0)

    def test_numeric_partial_credit_does_not_unlock_words(self):
        text = TEXT.replace('<answer>A</answer>', '<answer>120</answer>')
        result = score_response(text, '100', task_type='numeric', config=self.config())
        self.assertAlmostEqual(result['answer'], .6)
        self.assertEqual(result['words'], 0)


if __name__ == '__main__':
    unittest.main()
