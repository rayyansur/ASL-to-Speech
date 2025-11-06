import math

def normalize(landmarks):
    base_x, base_y, base_z = landmarks[0]
    norm_landmarks = []

    for x, y, z in landmarks:
        norm_landmarks.append((x - base_x, y - base_y, z - base_z))

    ref_x, ref_y, ref_z = norm_landmarks[9]
    scale = math.sqrt(ref_x ** 2 + ref_y ** 2 + ref_z ** 2)
    if scale == 0:
        scale = 1.0

    norm_landmarks = [(x / scale, y / scale, z / scale) for x, y, z in norm_landmarks]
    return norm_landmarks
