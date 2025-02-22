from drowsiness_detection.dataclasses import Eye
from drowsiness_detection.dataclasses.position import Position


def compute_eye_aspect_ratio(eye_position: list[Position]) -> float:
    """
    Compute the eye aspect ratio (EAR) for a given eye landmark. The EAR is a measure of how open the eye is.

    :param eye_position: The eye landmark positions.
    :type eye_position: list[Position]
    :return: The Eye Aspect Ratio (EAR) for the given eye landmark.
    :rtype: float
    :raises ValueError: If the eye landmark is not an instance of Eye.

    Ref: https://medium.com/analytics-vidhya/eye-aspect-ratio-ear-and-drowsiness-detector-using-dlib-a0b2c292d706
    """
    antecendents = abs(eye_position[1] - eye_position[5]) + abs(eye_position[2] - eye_position[4])
    consequents = 2 * abs(eye_position[0] - eye_position[3])
    ear = antecendents / consequents
    return ear
