import copy

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

from drowsiness_detection.dataclasses.drowsiness_image import DrowsinessImage

from .base_face_detector import BaseFaceDetector


class MediapipeFaceDetector(BaseFaceDetector):
    def __init__(self, face_detector_model_path: str) -> None:
        """
        Initializes the MediapipeFaceDetector object.

        :param face_detector_model_path: The path to the face detector model.
        :type face_detector_model_path: str
        """
        # Create an FaceDetector object.
        base_options = python.BaseOptions(model_asset_path=face_detector_model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=RunningMode.IMAGE)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, drowsiness_image: DrowsinessImage, inplace: bool = False) -> DrowsinessImage:
        """
        Detects the face in the image. This will add

        :param drowsiness_image: The image to detect the face.
        :type drowsiness_image: DrowsinessImage
        :param inplace: If True, the image will be modified in place. Otherwise, a new image will be created.
        :type inplace: bool
        :return: The image with the face detected.
        :rtype: DrowsinessImage

        :raises ValueError: If the image is not provided.
        :raises ValueError: If no face is detected in the image.
        """
        if drowsiness_image.image is None:
            raise ValueError("The image is not provided.")

        # Copy the image if not inplace
        if not inplace:
            drowsiness_image = copy.deepcopy(drowsiness_image)

        # Create mp image frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=drowsiness_image.image)

        # Process the image
        face_detector_result = self.detector.detect(mp_image)
        if not face_detector_result.face_landmarks:
            raise ValueError("No face detected in the image.")

        face_landmark = face_detector_result.face_landmarks[0]

        # Get the bounding box of the face
        min_x, min_y, max_x, max_y = float("inf"), float("inf"), float("-inf"), float("-inf")
        h, w, _ = drowsiness_image.image.shape
        for landmark in face_landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        # Add the face to the image
        drowsiness_image.face_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
        drowsiness_image.face.image = drowsiness_image.image[min_y:max_y, min_x:max_x]

        return drowsiness_image
