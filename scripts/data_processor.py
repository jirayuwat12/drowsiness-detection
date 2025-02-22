import argparse
import os

import mediapipe as mp
import cv2
from tqdm import tqdm
import numpy as np
import yaml
import shutil
from drowsiness_detection.dataclasses.drowsiness_image import DrowsinessImage
from drowsiness_detection.dataclasses.position import Position
from mediapipe.tasks import python
import pickle
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode


def main(face_detector_model_path: str, input_vdo_folder: str, skip_frames: int, output_data_folder: str, delete_output_if_exists: bool) -> None:
    # Delete the output folder and its content if it exists
    if delete_output_if_exists and os.path.exists(output_data_folder):
        shutil.rmtree(output_data_folder)

    # Create the output folder
    os.makedirs(output_data_folder, exist_ok=True)

    # List all the video files in the input_vdo_folder
    vdo_files = [os.path.join(input_vdo_folder, file) for file in os.listdir(input_vdo_folder)]

    for vdo_file in vdo_files:
        extract_faces_from_video(face_detector_model_path, vdo_file, skip_frames, output_data_folder)

def extract_faces_from_video(face_detector_model_path: str, vdo_path: str, skip_frames: int, output_data_folder: str) -> None:
    # Create an FaceDetector object.
    base_options = python.BaseOptions(model_asset_path=face_detector_model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=RunningMode.VIDEO)
    with vision.FaceLandmarker.create_from_options(options) as detector:
        vdo = cv2.VideoCapture(vdo_path)
        fps = vdo.get(cv2.CAP_PROP_FPS)

        for frame_index in tqdm(range(int(vdo.get(cv2.CAP_PROP_FRAME_COUNT))), unit="frame", desc=f"Processing the '{os.path.basename(vdo_path)}'"):
            ret, frame = vdo.read()

            if frame_index % skip_frames != 0:
                continue

            # Create an ImageFrame object.
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.array(frame),
            )
            timestamp = frame_index/fps

            # Detect face from image
            face_detector_result = detector.detect_for_video(mp_image, mp.Timestamp.from_seconds(timestamp).value)
            face_landmark = face_detector_result.face_landmarks[0]

            # Create a DrowsinessImage object
            data = DrowsinessImage(image=np.array(frame))
            min_x, min_y, max_x, max_y = 1e10, 1e10, -1, -1
            w = frame.shape[1]
            h = frame.shape[0]
            for landmark in face_landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                min_x, min_y, max_x, max_y = min(min_x, x), min(min_y, y), max(max_x, x), max(max_y, y)
            data.face_bbox = (min_x, min_y, max_x-min_x, max_y-min_y)
            data.face.image = frame[min_y:max_y, min_x:max_x]

            # Extract the left and right eye images
            data.face.left_eye_bbox = (face_landmark[362].x * w, face_landmark[475].y * h, (face_landmark[263].x - face_landmark[362].x) * w, (face_landmark[477].y - face_landmark[475].y) * h)
            data.face.eye.left_eye_image = frame[int(data.face.left_eye_bbox[1]):int(data.face.left_eye_bbox[1]+data.face.left_eye_bbox[3]), int(data.face.left_eye_bbox[0]):int(data.face.left_eye_bbox[0]+data.face.left_eye_bbox[2])]
            data.face.right_eye_bbox = (face_landmark[33].x * w, face_landmark[470].y * h, (face_landmark[133].x - face_landmark[33].x) * w, (face_landmark[472].y - face_landmark[470].y) * h)
            data.face.eye.right_eye_image = frame[int(data.face.right_eye_bbox[1]):int(data.face.right_eye_bbox[1]+data.face.right_eye_bbox[3]), int(data.face.right_eye_bbox[0]):int(data.face.right_eye_bbox[0]+data.face.right_eye_bbox[2])]
            # Offset the eye bbox to the face bbox
            data.face.left_eye_bbox = (data.face.left_eye_bbox[0] - data.face_bbox[0], data.face.left_eye_bbox[1] - data.face_bbox[1], data.face.left_eye_bbox[2], data.face.left_eye_bbox[3])
            data.face.right_eye_bbox = (data.face.right_eye_bbox[0] - data.face_bbox[0], data.face.right_eye_bbox[1] - data.face_bbox[1], data.face.right_eye_bbox[2], data.face.right_eye_bbox[3])

            # Set eye position
            data.face.eye.left_eye_positions = [
                Position(face_landmark[362].x, face_landmark[362].y),
                Position(face_landmark[384].x, face_landmark[384].y),
                Position(face_landmark[387].x, face_landmark[387].y),
                Position(face_landmark[263].x, face_landmark[263].y),
                Position(face_landmark[373].x, face_landmark[373].y),
                Position(face_landmark[380].x, face_landmark[380].y),
            ]
            data.face.eye.right_eye_positions = [
                Position(face_landmark[33].x, face_landmark[33].y),
                Position(face_landmark[160].x, face_landmark[160].y),
                Position(face_landmark[158].x, face_landmark[158].y),
                Position(face_landmark[133].x, face_landmark[133].y),
                Position(face_landmark[153].x, face_landmark[153].y),
                Position(face_landmark[144].x, face_landmark[144].y),
            ]
            # Offset the eye positions to the face bbox
            for eye_position in data.face.eye.left_eye_positions:
                eye_position.x = eye_position.x * w - data.face_bbox[0]
                eye_position.y = eye_position.y * h - data.face_bbox[1]
            for eye_position in data.face.eye.right_eye_positions:
                eye_position.x = eye_position.x * w - data.face_bbox[0]
                eye_position.y = eye_position.y * h - data.face_bbox[1]

            # Save the DrowsinessImage object
            output_file = os.path.join(output_data_folder, f"{os.path.basename(vdo_path).split('.')[0]}_{frame_index}.pkl")
            with open(output_file, "wb") as file:
                pickle.dump(data, file)

        vdo.release()

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Process the data")

    # Add the arguments
    parser.add_argument("--config", help="The configuration file", default="./configs/data_processor.example.yaml")

    # Parse the arguments
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    main(face_detector_model_path=config["face_detector_model_path"], input_vdo_folder=config["input_vdo_folder"], skip_frames=config['skip_frames'], delete_output_if_exists=config['delete_output_if_exists'], output_data_folder=config['output_data_folder'])
