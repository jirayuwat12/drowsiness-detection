from dataclasses import dataclass, field

import numpy as np

from .face import Face


@dataclass
class DrowsinessImage:
    """
    Dataclass to store the information of the image
    """

    image: np.ndarray | None = None
    face_bbox: tuple[int, int, int, int] | None = None
    face: Face = field(default_factory=Face)
    drowsiness_detected: bool = False
