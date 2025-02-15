from dataclasses import dataclass, field

import numpy as np

from .eye import Eye


@dataclass
class Face:
    """
    Dataclass to store the information of the face
    """

    image: np.ndarray | None = None
    left_eye_bbox: tuple[int, int, int, int] | None = None
    right_eye_bbox: tuple[int, int, int, int] | None = None
    eye: Eye = field(default_factory=Eye)
