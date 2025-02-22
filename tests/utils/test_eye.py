import pytest

from drowsiness_detection.dataclasses import Eye
from drowsiness_detection.dataclasses.position import Position
from drowsiness_detection.utils.eye import compute_eye_aspect_ratio


@pytest.fixture(scope="module")
def opened_eye_landmarks() -> Eye:
    return Eye(
        left_eye_positions=[
            Position(x=10, y=0),
            Position(x=11, y=3),
            Position(x=12, y=3),
            Position(x=13, y=0),
            Position(x=12, y=-3),
            Position(x=11, y=-3),
        ],
        right_eye_positions=[
            Position(x=0, y=0),
            Position(x=1, y=3),
            Position(x=2, y=3),
            Position(x=3, y=0),
            Position(x=2, y=-3),
            Position(x=1, y=-3),
        ],
    )


@pytest.fixture(scope="module")
def closed_eye_landmarks() -> Eye:
    return Eye(
        left_eye_positions=[
            Position(x=10, y=0),
            Position(x=11, y=-0.5),
            Position(x=12, y=-0.5),
            Position(x=13, y=0),
            Position(x=12, y=-0.75),
            Position(x=11, y=-0.75),
        ],
        right_eye_positions=[
            Position(x=0, y=0),
            Position(x=1, y=0),
            Position(x=2, y=0),
            Position(x=3, y=0),
            Position(x=2, y=0),
            Position(x=1, y=0),
        ],
    )


def test_compute_eye_aspect_ratio(opened_eye_landmarks: Eye, closed_eye_landmarks: Eye) -> None:
    opened_ear = compute_eye_aspect_ratio(opened_eye_landmarks.left_eye_positions)
    closed_ear = compute_eye_aspect_ratio(closed_eye_landmarks.left_eye_positions)
    assert opened_ear > closed_ear
    assert opened_ear > 0.2
    assert closed_ear < 0.2


def test_invalid_eye_landmarks() -> None:
    with pytest.raises(ValueError, match="The eye landmark must be an instance of list\[Position\]"):
        compute_eye_aspect_ratio(Eye())

    with pytest.raises(ValueError, match="The eye landmark must be an instance of list\[Position\]"):
        compute_eye_aspect_ratio("invalid input")


def test_correct_eye_landmarks(opened_eye_landmarks: Eye) -> None:
    # Pass the eye landmarks class directly
    _ = compute_eye_aspect_ratio(opened_eye_landmarks.right_eye_positions)
