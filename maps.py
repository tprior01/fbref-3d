from sqlalchemy import Numeric

seasons = {
    "2022-23": 23,
    "2021-22": 22,
    "2020-21": 21,
    "2019-20": 20,
    "2018-19": 19,
    "2017-18": 18}

comps = {
    "Premier League": "ENG",
    "Serie A": "ITA", "Bundesliga": "GER",
    "La Liga": "SPA",
    "Ligue 1": "FRA",
    "Champions League": "CL",
    "Europa League": "EL"}

cat_options = {
    "Shooting": "shooting",
    "Passing": "passing",
    "Pass Types": "passing_types",
    "Shot Creation": "gca",
    "Possession": "possession",
    "Defense": "defense",
    "Miscellaneous": "misc",
    "Playing Time": "playingtime",
    "Keepers": "keepers",
    "Keepers Advanced": "keepersadv"
}

positions = [
    "Goalkeeper",
    "Centre-Back",
    "Full-Back",
    "Defensive Midfield",
    "Central Midfield",
    "Attacking Midfield",
    "Winger",
    "Forward"
]

not_per_min = {
    'matches_gk',
    'starts_gk',
    'minutes_gk',
    'nineties_gk',
    'matches',
    'minutes',
    'nineties',
    'starts',
    "minutes",
    "pk_save_perc",
    "launch_completion",
    "launch_perc",
    "goal_kick_launch_perc",
    "cross_stop_perc",
    "pass_completion",
    "pass_completion_short",
    "pass_completion_medium",
    "pass_completion_long",
    "dribbler_tackle_perc",
    "take_on_success",
    "aerial_success",
    "avg_gk_pass_len",
    "avg_goal_kick_len",
    "avg_def_outside_pen_dist",
    "goal_per_shot",
    "goal_per_sot",
    "npxg_per_shot",
    "avg_shot_dist",
}

aggregates = {
    "save_perc": ["saves", "sot_ag", 100, Numeric(4, 1)],
    "clean_sheet_perc": ["saves", "sot_ag", 100, Numeric(4, 1)],
    "pk_save_perc": ["pk_saved", "pk_att_ag", 100, Numeric(4, 1)],
    "launch_completion": ["cmp_launch", "att_launch", 100, Numeric(4, 1)],
    "goal_kick_launch_perc": ["goal_kicks_launched", "att_goal_kick", 100, Numeric(4, 1)],
    "cross_stop_perc": ["stop_cross_ag", "att_cross_ag", 100, Numeric(4, 1)],
    "pass_completion": ["cmp_pass", "att_pass", 100, Numeric(4, 1)],
    "pass_completion_short": ["cmp_pass_short", "att_pass_short", 100, Numeric(4, 1)],
    "pass_completion_medium": ["cmp_pass_medium", "att_pass_medium", 100, Numeric(4, 1)],
    "pass_completion_long": ["cmp_pass_long", "att_pass_long"],
    "dribbler_tackle_perc": ["cmp_drib_tackle", "att_drib_tackle"],
    "take_on_success": ["cmp_take_on", "att_take_on"],
    "avg_gk_pass_len": ["gk_pass_len", "att_gk_pass", 1, Numeric(3, 1)],
    "avg_goal_kick_len": ["goal_kick_len", "att_goal_kick", 1, Numeric(3, 1)],
    "avg_def_outside_pen_dist": ["def_outside_pen_dist", "def_outside_pen", 1, Numeric(3, 1)],
    "goal_per_shot":  ["goals", "shot", 1, Numeric(3, 2)],
    "goal_per_sot": ["goals", "shot_on_target", 1, Numeric(3, 2)],
    "npxg_per_shot": ["npxg", "shot", 1, Numeric(3, 2)],
    "avg_shot_dist": ["tot_shot_dist", "shot", 1, Numeric(4, 1)],
    "aerial_success": ["aerials_won", "aerials_attempted", 100, Numeric(4, 1)],
    "launch_perc": ["att_launch_non_goal_kick", "att_pass_non_goal_kick", 100, Numeric(4, 1)]
}

sub_options = {
    "shooting":
        {
            "Goals": "goals",
            "Shots": "shot",
            "Shots on Target": "shot_on_target",
            "Freekicks": "fk",
            "Penalties Scored": "pk",
            "Penalties Attempted": "att_pk",
            "xG": "xg",
            "npxG": "npxg",
            "Goals - xG": "goals_minus_xg",
            "npGoals - npxG": "non_penalty_goals_minus_npxg",
            "Goals / Shot": "goal_per_shot",
            "Goals / Shot on Target": "goal_per_sot",
            "npxG / Shot": "npxg_per_shot",
            "Average Shot Distance": "avg_shot_dist"
        },
    "passing":
        {
            'test'
        },
    "Pass Types":
        {

        },
    "Shot Creation":
        {

        },
    "Possession":
        {

        },
    "Defense":
        {

        },
    "Miscellaneous":
        {

        },
    "Playing Time":
        {

        },
    "Keepers":
        {

        },
    "Keepers Advanced":
        {

        }
}
