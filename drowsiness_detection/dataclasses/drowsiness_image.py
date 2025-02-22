from dataclasses import dataclass, field

import numpy as np

from .face import Face


@dataclass
class DrowsinessImage:
    """
    Dataclass to store the information of the image

    :param image: The image to store
    :type image: np.ndarray
    :param face_bbox: The bounding box of the face in the image (x, y, width, height)
    :type face_bbox: tuple[int, int, int, int]
    :param face: The face in the image
    :type face: Face
    :param drowsiness_detected: The flag to indicate if the drowsiness is detected in the image
    :type drowsiness_detected
    """

    image: np.ndarray | None = None
    face_bbox: tuple[int, int, int, int] | None = None
    face: Face = field(default_factory=Face)
    drowsiness_detected: bool = False
