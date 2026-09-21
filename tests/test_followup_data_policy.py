import unittest
from spatial_intelligence.followup_data_policy import validate_leakage_policy


class PolicyTests(unittest.TestCase):
    def test_clean_and_bounded_exception(self):
        self.assertEqual(validate_leakage_policy({'leakage_checked':True}), 'checked')
        value={'leakage_checked':False, 'known_sources_leakage_checked':True,
               'contamination_waiver':{'authorized':True, 'scope':'openspatial_only',
                  'scene_overlap':'unknown','validation_policy':'all OpenSpatial train-only',
                  'reason':'explicit user authorization'}}
        self.assertIn('unknown',validate_leakage_policy(value))
        self.assertFalse(value['leakage_checked'])
        for key,replacement in [('authorized',False),('scope','all_sources'),('scene_overlap',0),('validation_policy','random_rows')]:
            invalid={**value,'contamination_waiver':{**value['contamination_waiver'],key:replacement}}
            with self.assertRaises(ValueError):validate_leakage_policy(invalid)

    def test_missing_audit_never_passes(self):
        for value in ({},{'leakage_checked':False},{'known_sources_leakage_checked':True}):
            with self.assertRaises(ValueError):validate_leakage_policy(value)


if __name__=='__main__':unittest.main()
