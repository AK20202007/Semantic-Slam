import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

class MapViewer:
    def __init__(self, width=800, height=600):
        if not glfw.init():
            raise Exception("GLFW initialization failed")
            
        self.window = glfw.create_window(width, height, "Semantic SLAM 3D Viewer", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")
            
        glfw.make_context_current(self.window)
        glEnable(GL_DEPTH_TEST)
        glPointSize(3.0)
        
        # Simple static camera setup for viewing the map
        self.cam_pos = [0.0, -10.0, -20.0]
        
    def draw_axes(self):
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0); glVertex3f(0.0, 0.0, 0.0); glVertex3f(1.0, 0.0, 0.0) # X
        glColor3f(0.0, 1.0, 0.0); glVertex3f(0.0, 0.0, 0.0); glVertex3f(0.0, 1.0, 0.0) # Y
        glColor3f(0.0, 0.0, 1.0); glVertex3f(0.0, 0.0, 0.0); glVertex3f(0.0, 0.0, 1.0) # Z
        glEnd()

    def update(self, vo):
        if glfw.window_should_close(self.window):
            return False
            
        glfw.poll_events()
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Set camera looking at the origin
        gluLookAt(self.cam_pos[0], self.cam_pos[1], self.cam_pos[2], 
                  0, 0, 0, 
                  0, -1, 0)
                  
        self.draw_axes()
        
        # Draw Trajectory (Green)
        glColor3f(0.0, 1.0, 0.0)
        glBegin(GL_LINE_STRIP)
        for _, t in vo.trajectory:
            glVertex3f(t[0][0], t[1][0], t[2][0])
        glEnd()
        
        # Draw Point Cloud (White)
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_POINTS)
        for pt in vo.point_cloud:
            glVertex3f(pt[0][0], pt[1][0], pt[2][0])
        glEnd()
        
        # Draw semantic markers (Blue and large)
        glColor3f(0.0, 0.5, 1.0)
        glPointSize(8.0)
        glBegin(GL_POINTS)
        for label_info in vo.labels_3d:
            pt = label_info['pt']
            glVertex3f(pt[0][0], pt[1][0], pt[2][0])
        glEnd()
        glPointSize(3.0)
        
        # Draw current camera pose as a red point
        if vo.cur_t is not None:
            glColor3f(1.0, 0.0, 0.0)
            glPointSize(5.0)
            glBegin(GL_POINTS)
            glVertex3f(vo.cur_t[0][0], vo.cur_t[1][0], vo.cur_t[2][0])
            glEnd()
            glPointSize(3.0)

        glfw.swap_buffers(self.window)
        return True

    def stop(self):
        glfw.terminate()
