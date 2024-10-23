RACE_FEATURES = ["trackCode", "distance", "raceId", "meetingDate", "raceTypeCode", "raceNumber", "auth_code"]

DOG_FEATURES = [
    "dogId",
    "runId",
    "prizeMoneyTotal",
    "colourCode",
    "careerPrizeMoney",
    "incomingGrade",
    "dog_age_days",
    "weightInKg",
    "sex",
    "ownerId",
    "trainerId",
    "damId",
    "sireId",
    "rating",
    "rugNumber",
    "prizeMoney",
]

DOG_LAST_FEATURES = [
    "avg_weight_last_5",
    "max_weight_last_5",
    "min_weight_last_5",
    "var_weight_last_5",
    "avg_rating_last_5",
    "min_rating_last_5",
    "max_rating_last_5",
    "var_rating_last_5",
    "days_since_last_race",
]

BOX_FEATURES = [
    "boxNumber",
    "box_num_races_last_5",
    "box_wins_last_5",
    "box_top_3_finishes_last_5",
    "box_win_rate_last_5",
    "box_top_3_finishes_rate_last_5",
    "box_avg_finish_last_5",
]

FINISHES_FEATURES = [
    "avg_finish_last_5",
    "median_finish_last_5",
    "best_finish_last_5",
    "worst_finish_last_5",
    "var_finish_last_5",
    "avg_finish_last_3",
    "median_finish_last_3",
    "best_finish_last_3",
    "worst_finish_last_3",
    "var_finish_last_3",
    "win_rate_last_5",
    "win_rate_last_3",
    "top_3_rate_last_5",
    "top_3_rate_last_3",
    "performance_trend",
]

SPEED_FEATURES = [
    "avg_speed_last_5",
    "max_speed_last_5",
    "min_speed_last_5",
    "avg_speed_last_3",
    "max_speed_last_3",
    "min_speed_last_3",
]

MARGIN_FEATURES = [
    "loss_margin_avg_last_5",
    "loss_margin_max_last_5",
    "loss_margin_min_last_5",
    "loss_margin_consistency_last_5",
    "close_loss_rate_last_5",
    "close_loss_rate_loss_last_5",
    "win_margin_avg_last_5",
    "win_margin_max_last_5",
    "win_margin_min_last_5",
    "win_margin_consistency_last_5",
    "close_win_rate_last_5",
    "close_win_rate_wins_last_5",
]

TRACK_DISTANCE_FEATURES = [
    "track_distance_num_races_last_5",
    "track_distance_wins_last_5",
    "track_distance_win_rate_last_5",
    "track_distance_top_3_finishes_last_5",
    "track_distance_top_3_rate_last_5",
    "track_distance_avg_finish_last_5",
    "track_distance_median_finish_last_5",
    "track_distance_avg_time_last_5",
    "track_distance_best_time_last_5",
    "track_distance_worst_time_last_5",
    "distance_num_races_last_5",
    "distance_wins_last_5",
    "distance_win_rate_last_5",
    "distance_top_3_finishes_last_5",
    "distance_top_3_rate_last_5",
    "distance_avg_finish_last_5",
    "distance_avg_time_last_5",
    "distance_best_time_last_5",
    "distance_worst_time_last_5",
]

PIR_FEATURES = [
    "pir_avg",
    "pir_volatility",
    "pir_starting_avg",
    "pir_improvement_avg",
    "pir_best",
    "pir_worst",
    "pir_range",
    "pir_median",
    "pir_early_avg",
    "pir_late_avg",
    "pir_consistency",
    "pir_top3_ratio",
    "pir_trend",
    "pir_trend_per_race_avg",
]

FORM_STATISTICS_FEATURES = [
    "form_statistics_sprint_third",
    "form_statistics_main_starts",
    "form_statistics_main_first",
    "form_statistics_main_second",
    "form_statistics_main_third",
    "form_statistics_staying_starts",
    "form_statistics_staying_first",
    "form_statistics_staying_second",
    "form_statistics_staying_third",
    "form_statistics_track_starts",
    "form_statistics_track_first",
    "form_statistics_track_second",
    "form_statistics_track_third",
    "form_statistics_distance_starts",
    "form_statistics_distance_first",
    "form_statistics_distance_second",
    "form_statistics_distance_third",
    "form_statistics_box_starts",
    "form_statistics_box_first",
    "form_statistics_box_second",
    "form_statistics_box_third",
    "form_statistics_trackDistance_starts",
    "form_statistics_trackDistance_first",
    "form_statistics_trackDistance_second",
    "form_statistics_trackDistance_third",
    "form_statistics_trackDistanceBox_starts",
    "form_statistics_trackDistanceBox_first",
    "form_statistics_trackDistanceBox_second",
    "form_statistics_trackDistanceBox_third",
    "form_statistics_trackBox_starts",
    "form_statistics_trackBox_first",
    "form_statistics_trackBox_second",
    "form_statistics_trackBox_third",
    "form_statistics_trackType_starts",
    "form_statistics_trackType_first",
    "form_statistics_trackType_second",
    "form_statistics_trackType_third",
    "form_statistics_trackTypeBox_starts",
    "form_statistics_trackTypeBox_first",
    "form_statistics_trackTypeBox_second",
    "form_statistics_trackTypeBox_third",
]

FORM_SUMMARY_FEATURES = [
    "form_summary_bestFinishTrackDistance",
    "form_summary_bestFirstSplitTrackDistance",
    "form_summary_avgFirstSplitTrackDistance",
    "form_summary_bestFirstSplitTrackDistanceBox",
]

MEETING_DATE_FEATURES = [
    "meetingYear",
    "meetingMonth",
    "meetingWeek",
    "meetingDay",
    "meetingDayofweek",
    "meetingDayofyear",
    "meetingIs_month_end",
    "meetingIs_month_start",
    "meetingIs_quarter_end",
    "meetingIs_quarter_start",
    "meetingIs_year_end",
    "meetingIs_year_start",
    "meetingElapsed",
]

PREDICT_FEATURES = [
    "startPrice",
    "resultTime",
    "place",
]

ALL_FEATURES = (
    RACE_FEATURES
    + DOG_FEATURES
    + DOG_LAST_FEATURES
    + BOX_FEATURES
    + FINISHES_FEATURES
    + SPEED_FEATURES
    + MARGIN_FEATURES
    + TRACK_DISTANCE_FEATURES
    + PIR_FEATURES
    + FORM_STATISTICS_FEATURES
    + FORM_STATISTICS_FEATURES
    + MEETING_DATE_FEATURES
    + PREDICT_FEATURES
)
