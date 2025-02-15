from drowsiness_detection.dataclasses import Eye


def compute_eye_aspect_ratio(eye: Eye) -> float:
    """
    Compute the eye aspect ratio (EAR) for a given eye landmark. The EAR is a measure of how open the eye is.

    :param eye: The eye landmark to compute the EAR for.
    :type eye: Eye
    :return: The Eye Aspect Ratio (EAR) for the given eye landmark.
    :rtype: float
    :raises ValueError: If the eye landmark is not an instance of Eye.

    Ref: https://medium.com/analytics-vidhya/eye-aspect-ratio-ear-and-drowsiness-detector-using-dlib-a0b2c292d706
    """

    if not isinstance(eye, Eye):
        raise ValueError("The eye landmark must be an instance of Eye.")

    antecendents = abs(eye.left_eye_positions[1] - eye.left_eye_positions[5]) + abs(
        eye.left_eye_positions[2] - eye.left_eye_positions[4]
    )
    consequents = 2 * abs(eye.left_eye_positions[0] - eye.left_eye_positions[3])
    ear = antecendents / consequents
    return ear
