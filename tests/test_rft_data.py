import unittest
from pathlib import Path
import importlib.util
import tempfile
from spatial_intelligence.rft_data import (choices_dict, source_group, uniform_indices,
    group_split, filter_leakage, normalize_spatialladder, normalize_4drl)


class RFTDataTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('cv2'), 'OpenCV optional on coordinator')
    def test_real_cpu_video_decode_and_cache(self):
        import cv2
        import numpy as np
        from spatial_intelligence.rft_data import extract_video
        with tempfile.TemporaryDirectory() as tmp:
            video=Path(tmp)/'test.avi'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(32,32))
            for i in range(20): writer.write(np.full((32,32,3),i*10,dtype=np.uint8))
            writer.release()
            first=extract_video(video,Path(tmp)/'frames',8)
            self.assertEqual(first['decoded_frames'],20)
            self.assertEqual(first['frame_indices'],uniform_indices(20,8))
            self.assertEqual(len(first['media']),8)
            self.assertEqual(first,extract_video(video,Path(tmp)/'frames',8))
            with self.assertRaises(ValueError): extract_video(video,Path(tmp)/'too_many',32)

    def test_choices_strict(self):
        self.assertEqual(choices_dict(["A. hello", "B) there"]), {"A":"hello","B":"there"})
        with self.assertRaises(ValueError): choices_dict(["hello"])
        with self.assertRaises(ValueError): choices_dict(["A. x","A. y"])

    def test_source_groups(self):
        self.assertEqual(source_group("scene0001_00"), source_group("scene0001_02"))
        self.assertEqual(source_group("pjQV1lF4R48_24.mp4"), source_group("pjQV1lF4R48_25"))
        self.assertNotEqual(source_group("41069021"),source_group("41069025"))

    def test_temporal_sampling(self):
        indices=uniform_indices(300,8)
        self.assertEqual((indices[0],indices[-1],len(set(indices))), (0,299,8))
        with self.assertRaises(ValueError): uniform_indices(7,8)

    def test_scene_split_and_overlap(self):
        rows=[dict(id=str(i),source="x",scene_id=str(i//3),source_group=source_group(str(i//3))) for i in range(60)]
        split=group_split(rows)
        self.assertEqual(split,group_split(rows))
        train={x['source_group'] for x in split if x['split']=='train'}
        val={x['source_group'] for x in split if x['split']=='validation'}
        self.assertFalse(train & val)
        kept,excluded=filter_leakage(rows,{source_group('0')})
        self.assertEqual((len(kept),len(excluded)),(57,3))

    def test_spatial_preserves_released_order_no_fake_fps(self):
        row=dict(question_id=1,question="count?",answer="3",options=None,
                 image=["scene0001_00/0.jpg","scene0001_00/1035.jpg","scene0001_00/120.jpg"],
                 question_type="object count",data_type="video")
        result=normalize_spatialladder(row,"/media")
        self.assertEqual(result['original_frame_indices'],[0,1035,120])
        self.assertTrue(result['nonmonotonic_frame_order'])
        self.assertEqual(result['input_mode'],'images')
        self.assertIsNone(result['fps'])
        self.assertEqual(result['raw_answer'],'3')
        row['image'][0]='../escape.jpg'
        with self.assertRaises(ValueError): normalize_spatialladder(row,"/media")

    def test_4d_labels_not_cot_target(self):
        row=dict(Type='abs_dir',Question='Where?',A='left',B='right',Correct='B',
                 video_path='./raw/a.mp4',Answer_cot='reason')
        result=normalize_4drl(row,1,'/media')
        self.assertEqual(result['answer'],'B')
        self.assertEqual(result['answer_type'],'mcq')
        self.assertEqual(result['choices'],dict(A='left',B='right'))
        self.assertEqual(Path(result['video_path']),Path('/media/a.mp4'))
        self.assertEqual(result['raw_annotation'],row)
        del row['Type']
        row['A']='A. left'
        self.assertEqual(normalize_4drl(row,2,'/media')['choices']['A'],'left')
        row['Correct']='right'
        result=normalize_4drl(row,2,'/media')
        self.assertEqual((result['answer'],result['raw_answer']),('B','right'))
        row['Correct']='2'
        with self.assertRaises(ValueError): normalize_4drl(row,2,'/media')


if __name__ == '__main__': unittest.main()
