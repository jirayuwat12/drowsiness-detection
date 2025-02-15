from drowsiness_detection.dataclasses.position import Position


def test_position_size():
    pos = Position(3, 4)
    assert pos.size() == 5.0


def test_position_neg():
    pos = Position(3, 4)
    neg_pos = -pos
    assert neg_pos == Position(-3, -4)


def test_position_add():
    pos1 = Position(1, 2)
    pos2 = Position(3, 4)
    assert pos1 + pos2 == Position(4, 6)


def test_position_sub():
    pos1 = Position(5, 7)
    pos2 = Position(2, 3)
    assert pos1 - pos2 == Position(3, 4)


def test_position_mul():
    pos = Position(2, 3)
    assert pos * 2 == Position(4, 6)


def test_position_truediv():
    pos = Position(4, 6)
    assert pos / 2 == Position(2, 3)


def test_position_abs():
    pos = Position(3, 4)
    assert abs(pos) == 5.0


def test_position_eq():
    pos1 = Position(3, 4)
    pos2 = Position(3, 4)
    pos3 = Position(5, 6)
    assert pos1 == pos2
    assert pos1 != pos3
