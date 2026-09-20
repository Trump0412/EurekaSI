import unittest
from spatial_intelligence.geometry_rft_reward import parse_response, score_response, spatialladder_numeric_reward


def wrap(answer="A", observation="left right above", transition="move forward rotate", derivation="distance direction position"):
    return f"<think>Spatial Observation: {observation}\nSpatial Transition: {transition}\nAnswer Derivation: {derivation}</think><answer>{answer}</answer>"


class RewardTests(unittest.TestCase):
    def test_exact_total(self):
        result = score_response(wrap(), "A")
        self.assertEqual(result["total"], 1.55)
        self.assertEqual(score_response(wrap(), "B")["total"], 0.5)

    def test_gold_blind(self):
        a, b = score_response(wrap(), "A"), score_response(wrap(), "B")
        self.assertEqual(a["parsed_answer"], b["parsed_answer"])
        self.assertEqual(a["fields"], b["fields"])

    def test_conflicts_and_truncation(self):
        for text in (wrap() + "<answer>B</answer>", wrap("A or B"), "A", wrap() + "Actually B", "<think>broken<answer>A</answer>"):
            self.assertEqual(score_response(text, "A")["total"], 0)
        self.assertEqual(score_response(wrap(), "A", truncated=True)["total"], 0)

    def test_invalid_structure_can_score_answer(self):
        self.assertEqual(score_response("<answer>A</answer>", "A")["total"], 1)
        for text in (wrap(observation=""), wrap().replace("Spatial Transition:", "Answer Derivation:"), wrap().replace("Spatial Observation:", "Spatial observation:")):
            result = score_response(text, "A")
            self.assertEqual(result["structure"], 0)
            self.assertEqual(result["words"], 0)

    def test_distinct_word_boundaries_and_normalization(self):
        result = score_response(wrap(observation="LEFT-left_left; right! cleft upright", transition="MOVE move", derivation="distance distance"), "A")
        self.assertEqual(result["matches"]["Spatial Observation"], ["left", "right"])
        self.assertAlmostEqual(result["words"], 4 / 9)

    def test_numeric_source_thresholds_and_safety(self):
        self.assertEqual(spatialladder_numeric_reward(100, 100), 1)
        self.assertAlmostEqual(spatialladder_numeric_reward(120, 100), 0.6)
        self.assertEqual(spatialladder_numeric_reward(150, 100), 0.1)
        self.assertEqual(spatialladder_numeric_reward(0, 0), 1)
        self.assertEqual(spatialladder_numeric_reward(1, 0), 0)
        self.assertEqual(spatialladder_numeric_reward(-100, -100), 1)
        self.assertEqual(score_response(wrap("NaN"), "100", task_type="numeric")["answer"], 0)
        self.assertIsNone(parse_response(wrap("10 cm"), task_type="numeric")["parsed_answer"])

    def test_mcq_choice_and_invalid_gold(self):
        self.assertIsNone(parse_response(wrap("Z"), choices=["one", "two"])["parsed_answer"])
        with self.assertRaises(ValueError):
            score_response(wrap(), "A or B")

    def test_exact_mcq_wrapper_normalization_not_prose_extraction(self):
        for answer in ('A', 'a.', '(A)', '[ a ].', '( A ).'):
            self.assertEqual(score_response(wrap(answer), 'A')["answer"], 1)
            self.assertEqual(score_response(wrap(answer), 'B')["parsed_answer"], 'A')
        for answer in ('(A]', '[A', 'A or B', 'A because it is left', 'A. B.', '((A))'):
            self.assertIsNone(parse_response(wrap(answer))["parsed_answer"])

    def test_pinned_numeric_formula_boundary_parity(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy unavailable; run this parity case in the training environment")
        thresholds = np.linspace(.5, .95, int((.95 - .5) / .05 + 2))
        for target in (1., 10., 100.):
            predictions = [target * (1 + error) for error in np.linspace(0, .6, 121)]
            predictions += [target * (2 - threshold) for threshold in thresholds]
            for pred in predictions:
                official = (abs(pred - target) / target <= 1 - thresholds).mean()
                self.assertEqual(spatialladder_numeric_reward(pred, target), official)


if __name__ == "__main__":
    unittest.main()
