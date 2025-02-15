from dataclasses import dataclass


@dataclass
class Position:
    x: float
    y: float

    def size(self) -> float:
        return self.__abs__()

    def __neg__(self) -> "Position":
        return Position(-self.x, -self.y)

    def __add__(self, other: "Position") -> "Position":
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Position") -> "Position":
        return Position(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float) -> "Position":
        return Position(self.x * other, self.y * other)

    def __truediv__(self, other: float) -> "Position":
        return Position(self.x / other, self.y / other)

    def __abs__(self) -> float:
        """Return the Euclidean norm."""
        return (self.x**2 + self.y**2) ** 0.5

    def __eq__(self, other: "Position") -> bool:
        return self.x == other.x and self.y == other.y
