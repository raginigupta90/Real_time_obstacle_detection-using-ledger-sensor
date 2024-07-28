ef calculate_distance(center_x):
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
