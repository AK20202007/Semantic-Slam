import pytest
import os
import shutil
from cloud.map_sync import CloudSync

def test_cloud_sync_mock():
    # Ensure a clean state for testing
    cloud = CloudSync(mock=True)
    test_map_name = "test_map"
    
    # Fake data
    pt_cloud = [[0, 0, 1], [1, 1, 1]]
    labels = [{'pt': [0, 0, 1], 'label': 'desk'}]
    trajectory = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
    
    # Save map
    cloud.save_map(test_map_name, pt_cloud, labels, trajectory)
    
    # Assert file exists
    expected_path = os.path.join(cloud.local_dir, f"{test_map_name}.pkl")
    assert os.path.exists(expected_path)
    
    # Load map
    loaded_pt, loaded_labels, loaded_traj = cloud.load_map(test_map_name)
    assert len(loaded_pt) == 2
    assert len(loaded_labels) == 1
    assert loaded_labels[0]['label'] == 'desk'
    
    # Cleanup
    if os.path.exists(expected_path):
        os.remove(expected_path)
