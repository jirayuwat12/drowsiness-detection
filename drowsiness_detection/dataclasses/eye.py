from dataclasses import dataclass, field

import numpy as np

from drowsiness_detection.dataclasses.position import Position


@dataclass
class Eye:
    """
    Dataclass to store the information of the eyes

    :param left_eye_image: The image of the left eye.
    :type left_eye_image: np.ndarray
    :param right_eye_image: The image of the right eye.
    :type right_eye_image: np.ndarray
    :param left_eye_bbox: The bounding box of the left eye.
    :type left_eye_bbox: tuple[int, int, int, int]
    :param right_eye_bbox: The bounding box of the right eye.
    :type right_eye_bbox: tuple[int, int, int, int]
    :param left_eye_positions: The landmarks of the left eye.
    :type left_eye_positions: list[Position]
    :param right_eye_positions: The landmarks of the right eye.
    :type right_eye_positions: list[Position]
    """

    left_eye_image: np.ndarray | None = None
    right_eye_image: np.ndarray | None = None
    left_eye_positions: list[Position] | list[tuple[int, int]] = field(default_factory=list)
    right_eye_positions: list[Position] | list[tuple[int, int]] = field(default_factory=list)
