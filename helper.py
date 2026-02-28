import cv2
import mediapipe as mp
import numpy as np
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import statistics
from mediapipe.framework.formats import landmark_pb2
import time
from collections import deque
import statistics
from mediapipe.framework.formats import landmark_pb2

def plot_single_hand(landmarks_flat, label_name="Unknown"):
    """
    Plots a single hand from a flattened array of landmarks.
    """
    landmarks = landmarks_flat.reshape(21, 3) 
    x = landmarks[:, 0]
    y = landmarks[:, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=50, c='red', label='Joints')

    connections = [
        [0,1],[1,2],[2,3],[3,4],          # Thumb
        [0,5],[5,6],[6,7],[7,8],          # Index
        [0,9],[9,10],[10,11],[11,12],     # Middle
        [0,13],[13,14],[14,15],[15,16],   # Ring
        [0,17],[17,18],[18,19],[19,20],   # Pinky
        [0,5],[5,9],[9,13],[13,17],[0,17] # Palm Base
    ]

    for p1, p2 in connections:
        plt.plot([x[p1], x[p2]], [y[p1], y[p2]], 'k-', lw=2)

    plt.title(f"Gesture: {label_name}")
    plt.gca().invert_yaxis()
    plt.xlabel('x coordinates')
    plt.ylabel('y coordinates')
    plt.grid(True)
    plt.legend()
    plt.show()


def preprocess_landmarks(row):
    """
    Custom normalization for HaGRID/MediaPipe landmarks.
    """
    landmarks = np.array(row, dtype=np.float32).reshape(21, 3)
    
    # Recenter around wrist
    
    # Recenter around wrist
    wrist_xy = landmarks[0, :2] 
    landmarks[:, :2] = landmarks[:, :2] - wrist_xy
    
    # Scale based on middle finger tip distance
    # Scale based on middle finger tip distance
    wrist_point = landmarks[0, :2]
    mid_tip_point = landmarks[12, :2]
    
    scale_factor = np.linalg.norm(mid_tip_point - wrist_point)
    if scale_factor < 1e-6:
        scale_factor = 1.0
        
    landmarks[:, :2] = landmarks[:, :2] / scale_factor
    return landmarks.flatten()


class RealTimeGesturePredictor:
    def __init__(self, model, class_names, model_asset_path='hand_landmarker.task'):
        """
        Initializes the predictor with the modern MediaPipe Tasks API and a smoothing buffer.
        """
        self.model = model
        self.class_names = class_names
        
        # Buffer to stabilize the predictions on screen
        self.prediction_buffer = deque(maxlen=10)
        
        # Safe timestamp tracker for the Tasks API
        self.last_timestamp_ms = 0
        
        # Setup MediaPipe Tasks Vision API
        BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path),
            running_mode=VisionRunningMode.VIDEO, 
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.landmarker = self.HandLandmarker.create_from_options(options)
        self.mp_drawing = mp.solutions.drawing_utils  # type: ignore
        self.mp_hands_connections = mp.solutions.hands.HAND_CONNECTIONS  # type: ignore

    def start_stream(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return

        window_name = 'Real-Time Gesture Recognition'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print("Starting video stream... Press 'q' to exit.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Safely calculate strictly increasing timestamp
            current_timestamp_ms = int(time.time() * 1000)
            if current_timestamp_ms <= self.last_timestamp_ms:
                current_timestamp_ms = self.last_timestamp_ms + 1
            self.last_timestamp_ms = current_timestamp_ms
            
            # Detect landmarks
            detection_result = self.landmarker.detect_for_video(mp_image, current_timestamp_ms)
            
            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    
                    # 1. Extract raw coordinates
                    image_height, image_width, _ = frame.shape
                    
                    extracted_landmarks = []
                    for lm in hand_landmarks:
                        extracted_landmarks.extend([lm.x * image_width, lm.y * image_height, lm.z])
                        
                    # 2. Draw landmarks
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList() # type: ignore
                    hand_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)  # type: ignore
                        for landmark in hand_landmarks
                    ])
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks_proto, self.mp_hands_connections)
                    
                    # 3. Normalize and Predict
                    normalized_data = preprocess_landmarks(extracted_landmarks)
                    prediction = self.model.predict(normalized_data.reshape(1, -1))
                    
                    if isinstance(prediction[0], (int, np.integer)):
                        raw_gesture = self.class_names[prediction[0]]
                    else:
                        raw_gesture = str(prediction[0])
                        
                    # 4. Smooth the prediction using the buffer
                    self.prediction_buffer.append(raw_gesture)
                    smoothed_gesture = statistics.mode(self.prediction_buffer)
                        
                    # 5. Overlay text
                    cv2.putText(frame, f'Gesture: {smoothed_gesture}', (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                # Clear buffer if no hand is detected to avoid stale predictions
                self.prediction_buffer.clear()
            
            cv2.imshow(window_name, frame)
            if cv2.waitKey(5) & 0xFF == ord('q') :
                break
                
        cap.release()
        cv2.destroyAllWindows()