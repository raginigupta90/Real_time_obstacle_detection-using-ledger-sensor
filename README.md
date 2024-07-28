3D Obstacle Detection and Distance Measurement using Stereo Vision

This project implements a real-time 3D obstacle detection and distance measurement system using stereo vision with OpenCV. The system uses two webcams to capture stereo images, detects obstacles in the environment, and calculates their distance from the cameras.

Features
Real-time Stereo Vision: Utilizes two webcams to capture left and right images simultaneously.
3D Obstacle Detection: Detects obstacles in the captured images and marks them with bounding boxes.
Distance Measurement: Calculates the distance of each detected obstacle from the cameras.
Visualization: Draws bounding boxes around detected obstacles and lines from the sensor to each obstacle, displaying the calculated distance.

Prerequisites
Python 3.x
OpenCV
NumPy

Installation
Clone the repository:
git clone https://github.com/yourusername/3d-obstacle-detection.git
cd 3d-obstacle-detection

Install the required packages:
pip install opencv-python numpy

Usage
Connect two webcams to your computer.
Update the focal_length and sensor_width parameters in sensor.py with the correct values for your cameras.

Run the script:
python sensor.py

The script will open a window displaying the video feed with detected obstacles and their distances.

Code Overview
sensor.py: The main script that captures video from the webcams, detects obstacles, and calculates distances.
calculate_distance(center_x): Calculates the distance to an object based on its position in the image.
detect_obstacles(frame_left, frame_right): Detects obstacles in the frame and calculates their distances.
draw_sensor(frame): Draws the sensor at the center of the frame.


Troubleshooting
If you encounter issues with the camera, ensure that the camera indices in cv2.VideoCapture() are correct.
If the script fails to capture frames, try using a different backend for video capture (e.g., cv2.CAP_DSHOW).


Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

License
This project is licensed under the MIT License. See the LICENSE file for details.

