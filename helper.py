import matplotlib.pyplot as plt
import numpy as np


def plot_single_hand(landmarks_flat, label_name="Unknown"):
    """
    Plots a single hand from a flattened array of landmarks.
    """
    # 1. Reshape the flat array (63 values) into (21 points, 3 coords)
    
    landmarks = landmarks_flat.reshape(21, 3) 
    
    x = landmarks[:, 0]
    y = landmarks[:, 1]

    plt.figure(figsize=(6, 6))
    
    # 2. Plot the Points (Joints)
    plt.scatter(x, y, s=50, c='red', label='Joints')

    # 3. Define the Connections (MediaPipe Standard)
    connections = [
        [0,1],[1,2],[2,3],[3,4],          # Thumb
        [0,5],[5,6],[6,7],[7,8],          # Index
        [0,9],[9,10],[10,11],[11,12],     # Middle
        [0,13],[13,14],[14,15],[15,16],   # Ring
        [0,17],[17,18],[18,19],[19,20],   # Pinky
        [0,5],[5,9],[9,13],[13,17],[0,17] # Palm Base
    ]

    # 4. Draw Lines (Bones)
    for p1, p2 in connections:
        plt.plot([x[p1], x[p2]], [y[p1], y[p2]], 'k-', lw=2)

    # Formatting
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
    Input: A single flat row of 63 values (21 landmarks * 3 coords).
    Output: Normalized flat row.
    """
    # 1. Reshape to (21, 3) for geometric manipulation
    landmarks = np.array(row, dtype=np.float32).reshape(21, 3)

    # 2. Recenter: Make Wrist (index 0) the origin (0,0)
    wrist_xy = landmarks[0, :2] 
    landmarks[:, :2] = landmarks[:, :2] - wrist_xy
    
    # 3. Scale: Normalize by the length of the Middle Finger
    wrist_point = landmarks[0, :2]
    mid_tip_point = landmarks[12, :2]
    
    # Calculate Euclidean distance between Wrist and Mid-Tip in 2D
    scale_factor = np.linalg.norm(mid_tip_point - wrist_point)
    
    # Avoid division by zero 
    if scale_factor < 1e-6:
        scale_factor = 1.0
        
    # Apply scaling to X and Y
    landmarks[:, :2] = landmarks[:, :2] / scale_factor
    
    # Flatten back to a 1D array
    return landmarks.flatten()


