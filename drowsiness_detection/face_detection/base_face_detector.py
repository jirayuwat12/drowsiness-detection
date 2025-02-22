from abc import ABC, abstractmethod

from drowsiness_detection.dataclasses.drowsiness_image import DrowsinessImage


class BaseFaceDetector(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def detect(self, drowsiness_image: DrowsinessImage, inplace: bool = False) -> DrowsinessImage:
        """
        Detects the face in the image

        :param drowsiness_image: The image to detect the face.
        :type drowsiness_image: DrowsinessImage
        :param inplace: If True, the image will be modified in place. Otherwise, a new image will be created.
        :type inplace: bool
        :return: The image with the face detected.
        :rtype: DrowsinessImage
        """
        pass
