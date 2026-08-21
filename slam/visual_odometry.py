import cv2
import numpy as np

class VisualOdometry:
    def __init__(self, camera_matrix):
        self.frame_stage = 0
        self.cam = camera_matrix
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.focal = camera_matrix[0, 0]
        self.pp = (camera_matrix[0, 2], camera_matrix[1, 2])
        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        
        # We will keep a map of 3D points
        self.point_cloud = []
        self.trajectory = []
        self.labels_3d = [] # List of dicts: {'pt': (x,y,z), 'label': str}
        
    def getAbsoluteScale(self):
        # For a monocular system without IMU or known patterns, scale is ambiguous.
        # We assume a constant scale of 1.0 for this MVP to simulate movement.
        return 1.0

    def process_first_frame(self):
        self.px_ref = self.detector.detect(self.new_frame)
        if len(self.px_ref) == 0:
            return
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        self.frame_stage = 1

    def process_second_frame(self):
        if self.px_ref is None or len(self.px_ref) == 0:
            self.process_first_frame()
            return
            
        self.px_ref, self.px_cur = self.featureTracking(self.last_frame, self.new_frame, self.px_ref)
        if len(self.px_ref) < 8:
            return
            
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            return
            
        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp=self.pp)
        self.frame_stage = 2
        self.px_ref = self.px_cur
        
        self.trajectory.append((self.cur_R.copy(), self.cur_t.copy()))

    def process_frame(self):
        self.px_ref, self.px_cur = self.featureTracking(self.last_frame, self.new_frame, self.px_ref)
        
        if len(self.px_ref) < 100:
            new_features = self.detector.detect(self.last_frame)
            if len(new_features) > 0:
                new_features = np.array([x.pt for x in new_features], dtype=np.float32)
                self.px_ref = np.vstack((self.px_ref, new_features))
                self.px_ref, self.px_cur = self.featureTracking(self.last_frame, self.new_frame, self.px_ref)
            
        if len(self.px_ref) < 8:
            return

        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is not None and E.shape == (3, 3):
            _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp=self.pp)
            absolute_scale = self.getAbsoluteScale()
            if absolute_scale > 0.1:
                self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
                self.cur_R = R.dot(self.cur_R)
                
            self.trajectory.append((self.cur_R.copy(), self.cur_t.copy()))
            
            # Simple heuristic for map points for MVP visualization:
            # We add a point somewhat in front of the camera based on features
            # In a real SLAM this would be rigorous triangulation and bundle adjustment
            if len(self.px_cur) > 0:
                # Add one representative point per frame to point cloud for viz
                # This is just to demonstrate the system building a map
                pt = self.cur_t + self.cur_R.dot(np.array([[0], [0], [5.0]])) # 5 units in front
                self.point_cloud.append(pt)

        self.px_ref = self.px_cur

    def update(self, frame):
        if len(frame.shape) == 3:
            self.new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            self.new_frame = frame
            
        if self.frame_stage == 0:
            self.process_first_frame()
        elif self.frame_stage == 1:
            self.process_second_frame()
        elif self.frame_stage == 2:
            self.process_frame()
            
        self.last_frame = self.new_frame
        
    def featureTracking(self, image_ref, image_cur, px_ref):
        kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **dict(winSize=(21, 21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)))
        
        if kp2 is None or len(kp2) == 0:
            return np.array([]), np.array([])
            
        st = st.reshape(st.shape[0])
        kp1 = px_ref[st == 1]
        kp2 = kp2[st == 1]

        return kp1, kp2

    def add_semantic_label(self, label, bbox):
        # We associate a semantic label from 2D bbox to a 3D point in the map.
        # For this MVP, we just add a semantic label attached to the current camera pose
        if self.cur_t is not None:
            label_pt = self.cur_t + self.cur_R.dot(np.array([[0], [0], [3.0]]))
            self.labels_3d.append({'pt': label_pt, 'label': label})
