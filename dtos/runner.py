import datetime


class Runner:
    id: int
    name: str
    handical: float
    removal_date: datetime
    sort_priority: int

    def __init__(self, id: int, name: str, handicap: float, removal_date: datetime):
        self.id = id
        self.name = name
        self.handicap = handicap
        self.removal_date = removal_date
