import copy

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

from drowsiness_detection.dataclasses.face import Face
from drowsiness_detection.dataclasses.position import Position

from .base_eye_detector import BaseEyeDetector


class MediapipeEyeDetector(BaseEyeDetector):
    def __init__(self, face_detector_model_path: str) -> None:
        """
        Initializes the MediapipeEyeDetector object.

        :param face_detector_model_path: The path to the face detector model.
        :type face_detector_model_path: str
        """
        # Create an FaceDetector object.
        base_options = python.BaseOptions(model_asset_path=face_detector_model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=RunningMode.IMAGE)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, face: Face, inplace: bool = False) -> Face:
        """
        Detects the eyes in the face

        :param face: The face to detect the eyes.
        :type face: Face
        :param inplace: If True, the face will be modified in place. Otherwise, a new face will be returned.
        :type inplace: bool
        :return: The face with the eyes detected.
        :rtype: Face

        :raises ValueError: If the image is not provided.
        :raises ValueError: If no face is detected in the image.
        """
        if face.image is None:
            raise ValueError("The image is not provided.")

        # Copy the image if not inplace
        if not inplace:
            face = copy.deepcopy(face)

        # Create mp image frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face.image)
        h, w, _ = face.image.shape

        # Process the image
        face_detector_result = self.detector.detect(mp_image)
        if not face_detector_result.face_landmarks:
            raise ValueError("No face detected in the image.")
        face_landmark = face_detector_result.face_landmarks[0]

        # Extract the left and right eye images
        face.left_eye_bbox = (
            face_landmark[362].x * w,
            face_landmark[475].y * h,
            (face_landmark[263].x - face_landmark[362].x) * w,
            (face_landmark[477].y - face_landmark[475].y) * h,
        )
        face.eye.left_eye_image = face.image[
            int(face.left_eye_bbox[1]) : int(face.left_eye_bbox[1] + face.left_eye_bbox[3]),
            int(face.left_eye_bbox[0]) : int(face.left_eye_bbox[0] + face.left_eye_bbox[2]),
        ]
        face.right_eye_bbox = (
            face_landmark[33].x * w,
            face_landmark[470].y * h,
            (face_landmark[133].x - face_landmark[33].x) * w,
            (face_landmark[472].y - face_landmark[470].y) * h,
        )
        face.eye.right_eye_image = face.image[
            int(face.right_eye_bbox[1]) : int(face.right_eye_bbox[1] + face.right_eye_bbox[3]),
            int(face.right_eye_bbox[0]) : int(face.right_eye_bbox[0] + face.right_eye_bbox[2]),
        ]

        # Set eye position
        face.eye.left_eye_positions = [
            Position(face_landmark[362].x, face_landmark[362].y),
            Position(face_landmark[384].x, face_landmark[384].y),
            Position(face_landmark[387].x, face_landmark[387].y),
            Position(face_landmark[263].x, face_landmark[263].y),
            Position(face_landmark[373].x, face_landmark[373].y),
            Position(face_landmark[380].x, face_landmark[380].y),
        ]
        face.eye.right_eye_positions = [
            Position(face_landmark[33].x, face_landmark[33].y),
            Position(face_landmark[160].x, face_landmark[160].y),
            Position(face_landmark[158].x, face_landmark[158].y),
            Position(face_landmark[133].x, face_landmark[133].y),
            Position(face_landmark[153].x, face_landmark[153].y),
            Position(face_landmark[144].x, face_landmark[144].y),
        ]

        return face
