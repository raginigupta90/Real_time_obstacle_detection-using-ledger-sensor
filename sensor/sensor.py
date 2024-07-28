# #obstacle detection (with the help of dotted line)
# import cv2
# import numpy as np

# def detect_obstacles(frame, num_rays=100, max_distance=10):
#     """
#     Detect obstacles in the frame and draw dotted boundary lines around them.

#     Parameters:
#     frame (numpy.ndarray): The image frame from the webcam.
#     num_rays (int): The number of rays to cast for detecting obstacles.
#     max_distance (int): The maximum distance to check for obstacles.

#     Returns:
#     numpy.ndarray: The processed frame with obstacle boundaries drawn.
#     """
#     height, width = frame.shape[:2]
#     center = (width // 2, height // 2)
#     scale = min(center) / max_distance

#     # Convert frame to grayscale and apply binary threshold
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

#     # Generate angles for rays
#     angles = np.linspace(-3 * np.pi / 4, 3 * np.pi / 4, num_rays)
#     obstacle_points = []

#     for angle in angles:
#         for dist in np.linspace(0, max_distance, 100):
#             x = int(center[0] + np.cos(angle) * dist * scale)
#             y = int(center[1] - np.sin(angle) * dist * scale)
#             if x < 0 or x >= width or y < 0 or y >= height or binary[y, x] == 255:
#                 obstacle_points.append((x, y))
#                 break

#     # Draw dotted lines between obstacle points
#     for i in range(len(obstacle_points) - 1):
#         draw_dotted_line(frame, obstacle_points[i], obstacle_points[i + 1], (255, 0, 0))

#     # Draw center point
#     cv2.circle(frame, center, 5, (0, 0, 0), -1)
#     return frame

# def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=5):
#     """
#     Draw a dotted line between two points.

#     Parameters:
#     img (numpy.ndarray): The image on which to draw the line.
#     pt1 (tuple): The starting point of the line.
#     pt2 (tuple): The ending point of the line.
#     color (tuple): The color of the line.
#     thickness (int): The thickness of the dots.
#     gap (int): The gap between dots.
#     """
#     dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
#     pts = []
#     for i in np.arange(0, dist, gap):
#         r = i / dist
#         x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
#         y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
#         pts.append((x, y))
#     for p in pts:
#         cv2.circle(img, p, thickness, color, -1)

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect obstacles and draw boundary lines
#     frame = detect_obstacles(frame)

#     # Display the resulting frame
#     cv2.imshow('Obstacle Detection', frame)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()







# #distance of the objects
# import cv2
# import numpy as np

# # Camera parameters (example values, replace with actual camera calibration)
# focal_length = 1000  # Focal length in pixels
# sensor_width = 24  # Sensor width in mm (for calibration)
# image_width, image_height = 640, 480  # Image dimensions

# def calculate_distance(center_x, focal_length):
#     """
#     Calculate distance from the sensor to an object given its position on the image plane.
#     """
#     # Check if center_x is zero to avoid division by zero
#     if center_x == 0:
#         return float('inf')  # Return a large value or handle appropriately

#     # Calculate distance using the formula: distance = (sensor_width * focal_length) / (2 * center_x)
#     distance = (sensor_width * focal_length) / (2 * center_x)
#     return distance

# def detect_obstacles(frame):
#     """
#     Detect obstacles in the frame and determine their distance from the sensor.
#     """
#     height, width = frame.shape[:2]
#     center = (width // 2, height // 2)

#     # Convert frame to grayscale and apply binary threshold
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

#     # Detect contours in the binary image
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for contour in contours:
#         # Compute the bounding box of the contour
#         x, y, w, h = cv2.boundingRect(contour)
        
#         # Calculate the center of the bounding box
#         center_x = x + w // 2

#         # Calculate distance from the sensor to the object
#         distance = calculate_distance(center_x, focal_length)

#         # Draw the bounding box and distance on the frame
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cv2.putText(frame, f'{distance:.2f}cm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#     return frame

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect obstacles and draw boundary lines
#     frame = detect_obstacles(frame)

#     # Display the resulting frame
#     cv2.imshow('Obstacle Detection', frame)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()



#EDGES DETECTION(SEGMENTATION)
# import cv2
# import numpy as np

# def detect_obstacles(frame):
#     # Convert frame to grayscale for edge detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Apply Canny edge detection
#     edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    
#     return edges

# def main():
#     cap = cv2.VideoCapture(0)  # Open webcam (0 is typically the default webcam)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture frame from webcam.")
#             break
        
#         # Detect edges (simulated obstacle detection)
#         edges = detect_obstacles(frame)
        
#         # Display the resulting frame
#         cv2.imshow('Obstacle Detection', edges)
        
#         # Exit on 'q' press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()




# import cv2
# import numpy as np

# # Camera parameters (example values, replace with actual camera calibration)
# focal_length = 1000  # Focal length in pixels
# sensor_width = 24  # Sensor width in mm (for calibration)

# def calculate_distance(center_x):
#     """
#     Calculate distance from the sensor to an object given its position on the image plane.
#     """
#     # Check if center_x is zero to avoid division by zero
#     if center_x == 0:
#         return float('inf')  # Return a large value or handle appropriately

#     # Calculate distance using the formula: distance = (sensor_width * focal_length) / (2 * center_x)
#     distance = (sensor_width * focal_length) / (2 * center_x)
#     return distance

# def detect_obstacles(frame):
#     """
#     Detect obstacles in the frame and determine their distance from the sensor.
#     """
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         center_x = x + w // 2

#         distance = calculate_distance(center_x)

#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cv2.putText(frame, f'{distance:.2f}cm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#     return frame

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = detect_obstacles(frame)

#     cv2.imshow('Obstacle Detection', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



##This code is for showing the sensor only without lining from sensor to the object and also it determines the distance
# import cv2
# import numpy as np

# # Camera parameters (example values, replace with actual camera calibration)
# focal_length = 100  # Focal length in pixels
# sensor_width = 24  # Sensor width in mm (for calibration)

# def calculate_distance(center_x):
#     """
#     Calculate distance from the sensor to an object given its position on the image plane.
#     """
#     # Check if center_x is zero to avoid division by zero
#     if center_x == 0:
#         return float('inf')  # Return a large value or handle appropriately

#     # Calculate distance using the formula: distance = (sensor_width * focal_length) / (2 * center_x)
#     distance = (sensor_width * focal_length) / (2 * center_x)
#     return distance

# def detect_obstacles(frame):
#     """
#     Detect obstacles in the frame and determine their distance from the sensor.
#     """
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         center_x = x + w // 2

#         distance = calculate_distance(center_x)

#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cv2.putText(frame, f'{distance:.2f}cm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#     return frame

# def draw_sensor(frame):
#     """
#     Draw the sensor at the center of the frame.
#     """
#     height, width = frame.shape[:2]
#     center = (width // 2, height // 2)
#     cv2.circle(frame, center, 5, (0, 255, 0), -1)
#     cv2.putText(frame, 'Sensor', (center[0] - 40, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = detect_obstacles(frame)
#     draw_sensor(frame)

#     cv2.imshow('Obstacle Detection', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




##this code is for showing the distance from sensor as well as the lines from the sensor
# import cv2
# import numpy as np

# # Camera parameters (example values, replace with actual camera calibration)
# focal_length = 100  # Focal length in pixels
# sensor_width = 24  # Sensor width in mm (for calibration)

# def calculate_distance(center_x):
#     """
#     Calculate distance from the sensor to an object given its position on the image plane.
#     """
#     # Check if center_x is zero to avoid division by zero
#     if center_x == 0:
#         return float('inf')  # Return a large value or handle appropriately

#     # Calculate distance using the formula: distance = (sensor_width * focal_length) / (2 * center_x)
#     distance = (sensor_width * focal_length) / (2 * center_x)
#     return distance

# def detect_obstacles(frame):
#     """
#     Detect obstacles in the frame and determine their distance from the sensor.
#     """
#     height, width = frame.shape[:2]
#     sensor_center = (width // 2, height // 2)

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         center_x = x + w // 2
#         center_y = y + h // 2

#         distance = calculate_distance(center_x)

#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cv2.putText(frame, f'{distance:2f}cm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # Draw line from the sensor center to the center of the obstacle
#         cv2.line(frame, sensor_center, (center_x, center_y), (0, 255, 0), 2)

#     return frame

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = detect_obstacles(frame)

#     cv2.imshow('Obstacle Detection', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()





import cv2
import numpy as np

# Camera parameters (example values, replace with actual camera calibration)
focal_length = 100  # Focal length in pixels
sensor_width = 24  # Sensor width in mm (for calibration)
baseline = 100  # Baseline (distance between two cameras in mm, for disparity calculation)

def calculate_distance(disparity):
    """
    Calculate distance from the sensor to an object using disparity map.
    """
    # Check if disparity is zero to avoid division by zero
    if disparity == 0:
        return float('inf')  # Return a large value or handle appropriately

    # Calculate distance using the formula: distance = baseline * focal_length / disparity
    distance = (baseline * focal_length) / disparity
    return distance

def detect_obstacles(frame_left, frame_right):
    """
    Detect obstacles in the frame and determine their distance from the sensor.
    """
    height, width = frame_left.shape[:2]
    sensor_center = (width // 2, height // 2)

    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # Stereo block matching to compute disparity map
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(gray_left, gray_right)

    # Normalize disparity map for better visualization
    disparity_normalized = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    contours, _ = cv2.findContours(disparity_normalized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2

        distance = calculate_distance(disparity_normalized[center_y, center_x])

        cv2.rectangle(frame_left, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame_left, f'{distance:.2f}cm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw line from the sensor center to the center of the obstacle
        cv2.line(frame_left, sensor_center, (center_x, center_y), (0, 255, 0), 2)

    return frame_left, disparity_normalized

# Initialize webcam (assuming stereo setup or two synchronized cameras)
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)  # Adjust index if needed

while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    if not ret_left or not ret_right:
        break

    frame_obstacles, disparity_map = detect_obstacles(frame_left, frame_right)

    # Display the frame with detected obstacles and disparity map
    cv2.imshow('Obstacle Detection', frame_obstacles)
    cv2.imshow('Disparity Map', disparity_map.astype(np.uint8))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
