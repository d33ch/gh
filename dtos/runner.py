from dataclasses import dataclass
import datetime


@dataclass
class Runner:
    id: int
    name: str
    handical: float
    removal_date: datetime
    sort_priority: int
