import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from g1_pipette_sim.video import VideoFrames


class VideoFramesTests(unittest.TestCase):
    def test_sequential_duplicate_and_backward_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'frames.mp4'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),30,(32,32))
            self.assertTrue(writer.isOpened())
            for value in [20,60,100,140,180]:writer.write(np.full((32,32,3),value,np.uint8))
            writer.release();reader=VideoFrames(path)
            try:
                first=reader.read(0).copy();third=reader.read(2).copy()
                self.assertGreater(float(third.mean()-first.mean()),60)
                np.testing.assert_array_equal(reader.read(2),third)
                np.testing.assert_array_equal(reader.read(0),first)
                with self.assertRaises(RuntimeError):reader.read(8)
            finally:reader.release()
