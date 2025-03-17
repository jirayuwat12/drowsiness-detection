import cv2
import numpy as np
import streamlit as st

from drowsiness_detection.drowsiness_detector import DrowsinessDetector
from drowsiness_detection.eye_detection import MediapipeEyeDetector
from drowsiness_detection.face_detection import MediapipeFaceDetector
from drowsiness_detection.utils.eye import compute_eye_aspect_ratio


def webcam_page():
    st.title("Drowsiness Detection")
    st.write("This page read the webcam feed and detects drowsiness in real-time.")

    # Load the pre-trained model
    mp_face_detector = MediapipeFaceDetector(face_detector_model_path="./models/face_landmarker.task")
    mp_eye_detector = MediapipeEyeDetector(face_detector_model_path="./models/face_landmarker.task")
    drowsiness_detector = DrowsinessDetector(face_detector=mp_face_detector, eye_detector=mp_eye_detector)

    # Open the webcam
    cap = cv2.VideoCapture(0)

    frame_window = st.image([])

    col = st.columns(2)
    left_ear_list = [None] * 100
    left_ear_chart = col[0].line_chart(left_ear_list, use_container_width=True)
    right_ear_list = [None] * 100
    right_ear_chart = col[1].line_chart(right_ear_list, use_container_width=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        drowsiness_image = drowsiness_detector.detect(np.array(frame), return_drowsiness_image=True)

        # Compute EAR
        if drowsiness_image.drowsiness_detected:
            left_ear = compute_eye_aspect_ratio(drowsiness_image.face.eye.left_eye_positions)
            right_ear = compute_eye_aspect_ratio(drowsiness_image.face.eye.right_eye_positions)
        else:
            left_ear = None
            right_ear = None

        # Display EAR
        left_ear_list = left_ear_list[-100:] + [left_ear]
        right_ear_list = right_ear_list[-100:] + [right_ear]
        left_ear_chart.line_chart(left_ear_list, x_label="Time", y_label="Left EAR")
        right_ear_chart.line_chart(right_ear_list, x_label="Time", y_label="Right EAR")

        # Draw the face bounding box
        image = drowsiness_image.image.copy()
        if drowsiness_image.face_bbox is not None:
            face_x, face_y, face_width, face_height = drowsiness_image.face_bbox
            cv2.rectangle(image, (face_x, face_y), (face_x + face_width, face_y + face_height), (0, 255, 0), 2)

        # Draw the eye landmarks
        if drowsiness_image.face.left_eye_bbox is not None:
            left_eye_x, left_eye_y, left_eye_width, left_eye_height = [
                int(i) for i in drowsiness_image.face.left_eye_bbox
            ]
            left_eye_x += face_x
            left_eye_y += face_y
            cv2.rectangle(
                image,
                (left_eye_x, left_eye_y),
                (left_eye_x + left_eye_width, left_eye_y + left_eye_height),
                (0, 0, 255),
                2,
            )

        if drowsiness_image.face.right_eye_bbox is not None:
            right_eye_x, right_eye_y, right_eye_width, right_eye_height = [
                int(i) for i in drowsiness_image.face.right_eye_bbox
            ]
            right_eye_x += face_x
            right_eye_y += face_y
            cv2.rectangle(
                image,
                (right_eye_x, right_eye_y),
                (right_eye_x + right_eye_width, right_eye_y + right_eye_height),
                (0, 0, 255),
                2,
            )

        # Display the frame
        frame_window.image(image, channels="BGR", caption="Video from your webcam")

    cap.release()


def video_input_page():
    pass


def main():
    st.set_page_config(page_title="Drowsiness Detection", page_icon="🚗")
    st.sidebar.title("We have 2 options:")
    page = st.sidebar.radio("", ["Webcam", "Video Input"])

    match page:
        case "Webcam":
            webcam_page()
        case "Video Input":
            video_input_page()


if __name__ == "__main__":
    main()
