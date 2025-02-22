from drowsiness_detection.dataclasses.drowsiness_image import DrowsinessImage
from drowsiness_detection.eye_detection import BaseEyeDetector
from drowsiness_detection.face_detection import BaseFaceDetector
from drowsiness_detection.utils.eye import compute_eye_aspect_ratio


class DrowsinessDetector:
    def __init__(
        self, face_detector: BaseFaceDetector | None = None, eye_detector: BaseEyeDetector | None = None
    ) -> None:
        """
        Initializes the DrowsinessDetector

        :param face_detector: The face detector to use.
        :type face_detector: FaceDetector
        :param eye_detector: The eye detector to use.
        :type eye_detector: EyeDetector

        :raises ValueError: If face_detector is not an instance of BaseFaceDetector.
        :raises ValueError: If eye_detector is not an instance of BaseEyeDetector.
        """
        if not isinstance(face_detector, BaseFaceDetector):
            raise ValueError(f"face_detector should be an instance of BaseFaceDetector, not {type(face_detector)}")
        if not isinstance(eye_detector, BaseEyeDetector):
            raise ValueError(f"eye_detector should be an instance of BaseEyeDetector, not {type(eye_detector)}")
        self.face_detector = face_detector
        self.eye_detector = eye_detector

    def detect(
        self, drowsiness_image: DrowsinessImage, return_drowsiness_image: bool = False, ear_threshold: float = 0.5
    ) -> DrowsinessImage | bool:
        """
        Detects the drowsiness in the image

        :param drowsiness_image: The image to detect the drowsiness.
        :type drowsiness_image: DrowsinessImage
        :param return_drowsiness_image: Whether to return the drowsiness image.
        :type return_drowsiness_image: bool
        :param ear_threshold: The EAR threshold to consider the eye closed.
        :type ear_threshold: float
        :return: The image with the drowsiness detected.
        :rtype: DrowsinessImage
        """
        drowsiness_image = self.face_detector.detect(drowsiness_image)
        drowsiness_image.face = self.eye_detector.detect(drowsiness_image.face)

        left_ear = compute_eye_aspect_ratio(drowsiness_image.face.eye.left_eye_positions)
        right_ear = compute_eye_aspect_ratio(drowsiness_image.face.eye.right_eye_positions)

        if left_ear < ear_threshold and right_ear < ear_threshold:
            drowsiness_image.drowsiness_detected = True

        if return_drowsiness_image:
            return drowsiness_image

        return drowsiness_image.drowsiness_detected
