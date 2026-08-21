import pytest
import numpy as np
import sys
from unittest.mock import patch, MagicMock

# Mock tensorflow to avoid import errors if not installed locally
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.keras'] = MagicMock()
sys.modules['tensorflow.keras.applications'] = MagicMock()
sys.modules['tensorflow.keras.applications.mobilenet_v2'] = MagicMock()

from semantic.object_detector import SemanticDetector

def test_semantic_detector():
    with patch('semantic.object_detector.MobileNetV2') as MockModel, \
         patch('semantic.object_detector.preprocess_input') as MockPreprocess, \
         patch('semantic.object_detector.decode_predictions') as MockDecode:
         
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = np.array([[0.9]])
        MockModel.return_value = mock_model_instance
        
        MockDecode.return_value = [[('n0000', 'mock_chair', 0.85)]]
        
        detector = SemanticDetector(threshold=0.5)
        
        # Create a dummy image frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        label, conf, bbox = detector.detect(dummy_frame)
        
        assert label == 'mock_chair'
        assert conf == 0.85
        assert bbox is not None
        assert len(bbox) == 4
