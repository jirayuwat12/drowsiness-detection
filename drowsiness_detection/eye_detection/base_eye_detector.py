from abc import ABC, abstractmethod

from drowsiness_detection.dataclasses.face import Face


class BaseEyeDetector(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def detect(self, face: Face, inplace: bool = False) -> Face:
        """
        Detects the eyes in the face

        :param face: The face to detect the eyes.
        :type face: Face
        :param inplace: If True, the face will be modified in place. Otherwise, a new face will be created.
        :type inplace: bool
        :return: The face with the eyes detected.
        :rtype: Face
        """
        pass
