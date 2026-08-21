import pytest
import numpy as np
from slam.visual_odometry import VisualOdometry

def test_visual_odometry_init():
    focal = 718.8560
    pp = (320, 240)
    K = np.array([[focal, 0, pp[0]],
                  [0, focal, pp[1]],
                  [0, 0, 1]])
    
    vo = VisualOdometry(K)
    assert vo.frame_stage == 0
    assert vo.point_cloud == []
    assert vo.trajectory == []
    assert vo.labels_3d == []
    assert vo.focal == focal
    assert vo.pp == pp

def test_add_semantic_label():
    focal = 718.8560
    K = np.array([[focal, 0, 320],
                  [0, focal, 240],
                  [0, 0, 1]])
    vo = VisualOdometry(K)
    vo.cur_t = np.array([[0.0], [0.0], [0.0]])
    vo.cur_R = np.eye(3)
    
    vo.add_semantic_label("chair", (10, 10, 100, 100))
    assert len(vo.labels_3d) == 1
    assert vo.labels_3d[0]['label'] == "chair"
