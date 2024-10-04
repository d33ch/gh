from datetime import datetime
import math
import re
import numpy as np
import pandas as pd


class Utils:
    @staticmethod
    def flatten_dict(d, parent_key="", sep="_"):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(Utils.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def time_to_components(time_str):
        try:
            time_obj = datetime.strptime(time_str, "%I:%M%p")
            hour = time_obj.hour
            minute = time_obj.minute
            return hour, minute
        except ValueError as e:
            print(f"Error parsing time string '{time_str}': {e}")
            return None, None

    @staticmethod
    def extract_numeric(value):
        if value is None:
            return -1
        if isinstance(value, str):
            # Use regular expression to find all numbers (including decimal point)
            match = re.search(r"\d+\.?\d*", value)
            if match:
                return float(match.group())
        elif isinstance(value, (int, float)):
            return value
        return -1

    @staticmethod
    def encode_race_result(result):
        if result.isdigit():
            return int(result)
        result_map = {
            "T": -1,  # TailedOff
            "D": -2,  # Disqualified
            "P": -3,  # PulledUp
            "F": -4,  # Fell
            "U": -5,  # Unseated
            "B": -6,  # BroughtDown
            "R": -7,  # RefusedToRace
            "O": -8,  # Any other non-finish reason
        }
        return result_map.get(result, 0)  # Default to 0 if unknown

    @staticmethod
    def split_and_encode_last_results(x):
        if pd.isna(x) or x == "":
            return [-1] * 5
        return [Utils.encode_race_result(i) for i in x.split("-") if i]

    @staticmethod
    def split_and_encode_pir(x):
        if pd.isna(x) or x == "":
            return "0"
        return "".join(map(str, [Utils.encode_race_result(i) for i in list(x) if i]))

    @staticmethod
    def encode_grade_feature(grade):
        if pd.isna(grade) or grade == "":
            return np.nan
        if grade == "Maiden":
            return 0
        try:
            return int(grade)
        except ValueError:
            return np.nan

    @staticmethod
    def encode_missing_value(boxNumber, value):
        if boxNumber is None:
            return value
        if isinstance(boxNumber, str):
            return value if boxNumber == "" else boxNumber
        if math.isnan(boxNumber):
            return value
        return boxNumber
