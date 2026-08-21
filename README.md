# Cloud-Assisted Semantic SLAM for Indoor AR Navigation

A comprehensive Python-based MVP that bridges four complex domains in a single project:
1. **SLAM (Simultaneous Localization and Mapping)**: A lightweight Visual Odometry pipeline built with OpenCV to track camera movement and reconstruct a 3D point cloud.
2. **Real-time 3D Visualization (OpenGL)**: A fast `PyOpenGL` renderer displaying the map and camera trajectory dynamically.
3. **Semantic Understanding (Deep Learning)**: Uses Keras and a pre-trained `MobileNetV2` model to detect objects and attach semantic labels to the 3D map points.
4. **Cloud Offloading**: Uses `boto3` (and a local mock mode) to save and load generated semantic maps, demonstrating cloud readiness for collaborative or multi-session AR.

## Installation

Ensure you are using Python 3.10+. Create a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

On Linux systems, you will also need to install the OpenGL and GLFW system dependencies:
```bash
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglfw3 libglfw3-dev
```

## Usage

You can run the full SLAM pipeline using a webcam or a pre-recorded video:

```bash
# Run with default webcam
python main.py

# Run with a video file
python main.py path/to/video.mp4
```

Press `q` in the OpenCV video window to exit and trigger the cloud save routine.

## Testing

Tests are written using `pytest`. You can run them via:
```bash
PYTHONPATH=. pytest tests/
```
