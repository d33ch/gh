from typing import List

from models.ladder_position import LadderPosition


class PriceLadder:
    def __init__(self, positions: List[LadderPosition]):
        self.positions = positions
