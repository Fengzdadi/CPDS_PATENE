from scipy.interpolate import Akima1DInterpolator
import numpy as np

def interpolate_low_confidence(data, confidence_threshold=0.9):
    """
    Interpolate keypoints with low confidence using Akima interpolation.
    
    Parameters:
    - data: List of keypoints sequences.
    - confidence_threshold: Threshold for confidence value below which interpolation is applied.
    
    Returns:
    - Interpolated data.
    """
    n = len(data)
    interpolated_data = data.copy()
    
    # Compute the threshold ti for each keypoint
    confidences = [point[2::3] for point in data]
    mean_confidences = np.mean(confidences, axis=0)
    ti_values = mean_confidences * confidence_threshold
    
    # Check each keypoint's confidence and interpolate if necessary
    for i, keypoint in enumerate(data):
        for j in range(0, len(keypoint), 3):  # Iterate over every 3 values (x, y, confidence)
            confidence = keypoint[j+2]
            if confidence < ti_values[j//3]:
                # Indices for interpolation
                start_idx = max(0, i-2)
                end_idx = min(n, i+3)
                
                # Akima interpolation for x and y values
                x_values = [point[j] for point in data[start_idx:end_idx]]
                y_values = [point[j+1] for point in data[start_idx:end_idx]]
                interpolator_x = Akima1DInterpolator(list(range(start_idx, end_idx)), x_values)
                interpolator_y = Akima1DInterpolator(list(range(start_idx, end_idx)), y_values)
                
                interpolated_data[i][j] = interpolator_x(i)
                interpolated_data[i][j+1] = interpolator_y(i)
                
    return interpolated_data

# Apply the function on the sample data to get the interpolated keypoints
# interpolated_sample_data = interpolate_low_confidence(data)

# interpolated_sample_data


def normalize_scale(data):
    """
    Normalize the scale of the keypoints based on the distance between keypoints 2 and 9.
    
    Parameters:
    - data: List of keypoints sequences.
    
    Returns:
    - Scaled data.
    """
    scaled_data = []
    for keypoints in data:
        # Extract the coordinates of keypoints 2 and 9
        x2, y2 = keypoints[2*3], keypoints[2*3 + 1]
        x9, y9 = keypoints[9*3], keypoints[9*3 + 1]
        
        # Calculate the distance between keypoints 2 and 9
        distance = ((x9 - x2) ** 2 + (y9 - y2) ** 2) ** 0.5
        
        # Scale all keypoints based on this distance
        scaled_keypoints = [coord / distance for coord in keypoints]
        scaled_data.append(scaled_keypoints)
    
    return scaled_data

scaled_data = normalize_scale(interpolated_sample_data)
scaled_data

import math

def rotate_data(data):
    """
    Rotate the keypoints such that the line connecting keypoints 2 and 9 is vertical.
    
    Parameters:
    - data: List of scaled keypoints sequences.
    
    Returns:
    - Rotated data.
    """
    rotated_data = []
    for keypoints in data:
        # Extract the coordinates of keypoints 2 and 9
        x2, y2 = keypoints[2*3], keypoints[2*3 + 1]
        x9, y9 = keypoints[9*3], keypoints[9*3 + 1]
        
        # Calculate the angle between the line connecting keypoints 2 and 9 and the vertical axis
        angle = math.atan2(y9 - y2, x9 - x2) - math.pi/2
        
        # Rotation matrix
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        
        # Apply rotation to each keypoint
        rotated_keypoints = []
        for i in range(0, len(keypoints), 3):
            x, y = keypoints[i], keypoints[i+1]
            rotated_x = x * cos_angle - y * sin_angle
            rotated_y = x * sin_angle + y * cos_angle
            rotated_keypoints.extend([rotated_x, rotated_y, keypoints[i+2]])  # Add the confidence value unchanged
        
        rotated_data.append(rotated_keypoints)
    
    return rotated_data

rotated_data = rotate_data(scaled_data)
rotated_data
