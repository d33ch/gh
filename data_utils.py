from typing import List, Tuple
from base.base_data_utils import BaseDataUtils

class DataUtils(BaseDataUtils):
    def as_str(self, val):
        return '%.2f' % val if (type(val) is float) or (type(val) is int) else val if type(val) is str else ''

    def get_min_positive(self, a: float, b: float):
        min = min(a, b)
        return min if min > 0 else max(a, b)

    def slice(self, l, n, attr):
        try:
            x = getattr(l[n], attr) 
        except:
            x = ''
        return(x)

    def pull_ladder(ladder, size = 5) -> List[Tuple[float, float]]:
        return [(step.price, step.size) for step in ladder[:size]] if len(ladder) > 0 else [(0, 0)]


