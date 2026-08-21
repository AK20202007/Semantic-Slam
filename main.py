import cv2
import numpy as np
import threading
import sys
from slam.visual_odometry import VisualOdometry
from semantic.object_detector import SemanticDetector
from viewer.opengl_viewer import MapViewer
from cloud.map_sync import CloudSync

def main():
    print("Initializing Cloud-Assisted Semantic SLAM...")
    # 1. Initialize Cloud Sync
    cloud = CloudSync(mock=True)
    
    # 2. Setup Camera
    # Use video file if provided, otherwise default to webcam (0)
    video_source = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Fallback to defaults if width/height are 0
    if width == 0 or height == 0:
        width, height = 640, 480
        
    # Simple camera intrinsics for MVP
    focal = 718.8560
    pp = (width/2, height/2)
    K = np.array([[focal, 0, pp[0]],
                  [0, focal, pp[1]],
                  [0, 0, 1]])
                  
    # 3. Initialize SLAM and Semantic modules
    vo = VisualOdometry(K)
    detector = SemanticDetector(threshold=0.3)
    
    # 4. Initialize Viewer
    try:
        viewer = MapViewer(800, 600)
    except Exception as e:
        print(f"Failed to start OpenGL viewer: {e}")
        return
    
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run SLAM
        vo.update(frame)
        
        # Run Semantic Detection (run every 15 frames to save compute)
        if frame_id % 15 == 0:
            label, conf, bbox = detector.detect(frame)
            if label:
                print(f"Detected {label} ({conf:.2f})")
                vo.add_semantic_label(label, bbox)
                
        # Draw bounding boxes and labels on 2D frame if available
        # Note: We just draw the most recent one detected for MVP visualization
        if frame_id % 15 != 0 and 'label' in locals() and label is not None:
            cv2.putText(frame, f"{label} {conf:.2f}", (bbox[0], bbox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
        # Show 2D OpenCV View
        cv2.imshow('Camera Feed & Semantics', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        # Update 3D Viewer
        if not viewer.update(vo):
            break
            
        frame_id += 1
        
    # Save Map to Cloud on exit
    print("\nShutting down... Saving current map to cloud.")
    cloud.save_map("latest_session", vo.point_cloud, vo.labels_3d, vo.trajectory)
    
    cap.release()
    cv2.destroyAllWindows()
    viewer.stop()

if __name__ == "__main__":
    main()
