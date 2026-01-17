import mediapipe as mp          # MediaPipe library for face landmark detection
import cv2                     # OpenCV for image processing
import numpy as np              # NumPy for numerical operations


class FaceLandmarks:
    def __init__(self, max_num_faces=10, static_image_mode=False):
        # Access MediaPipe FaceMesh solution
        self.mp_face_mesh = mp.solutions.face_mesh

        # Initialize FaceMesh with configurable parameters
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,  # True for images, False for video streams
            max_num_faces=max_num_faces,           # Maximum number of faces to detect
            refine_landmarks=True,                 # Enable iris and refined landmarks
            min_detection_confidence=0.5,          # Minimum confidence for face detection
            min_tracking_confidence=0.5            # Minimum confidence for face tracking
        )

    def get_facial_landmarks(self, frame):
        # If frame is invalid, return empty list
        if frame is None:
            return []

        # Get frame height and width
        height, width = frame.shape[:2]

        # Convert BGR image (OpenCV default) to RGB (MediaPipe requirement)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run face landmark detection
        result = self.face_mesh.process(frame_rgb)

        # If no faces are detected, return empty list
        if not result.multi_face_landmarks:
            return []

        # List to store landmarks for all detected faces
        all_faces_landmarks = []

        # Iterate over each detected face
        for face_landmarks in result.multi_face_landmarks:
            landmarks = []

            # FaceMesh provides 468 landmarks per face
            for i in range(468):
                pt = face_landmarks.landmark[i]

                # Convert normalized coordinates to pixel coordinates
                x = int(pt.x * width)
                y = int(pt.y * height)

                # Append landmark point
                landmarks.append([x, y])

            # Convert landmarks list to NumPy array
            all_faces_landmarks.append(np.array(landmarks, dtype=np.int32))

        # Return landmarks for all faces
        return all_faces_landmarks
