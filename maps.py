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
    "pass_completion_long": ["cmp_pass_long", "att_pass_long", 100, Numeric(4, 1)],
    "dribbler_tackle_perc": ["cmp_drib_tackle", "att_drib_tackle", 100, Numeric(4, 1)],
    "take_on_success": ["cmp_take_on", "att_take_on", 100, Numeric(4, 1)],
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

axis_options = {
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
            'Completed Passes': 'cmp_pass',
            'Attempted Passes': 'att_pass',
            'Pass Completion': 'pass_completion',
            'Total Pass Distance': 'tot_pass_dist',
            'Progressive Pass Distance': 'prog_pass_dist',
            'Completed Short Passes': 'cmp_pass_short',
            'Attempted Short Passes': 'att_pass_short',
            'Short Pass Completion (%)': 'pass_completion_short',
            'Completed Medium Passes': 'cmp_pass_medium',
            'Attempted Medium Passes': 'att_pass_medium',
            'Medium Pass Completion (%)': 'medium_completion_short',
            'Completed Long Passes': 'cmp_pass_long',
            'Attempted Long Passes': 'att_pass_long',
            'Long Pass Completion (%)': 'long_completion_short',
            'Assists': 'assist',
            'xAG': 'xag',
            'xA': 'xa',
            'Assists - xAG': 'assists_minus_xag',
            'Key Passes': 'key_pass',
            'Passes to Final Third': 'fin_3rd_pass',
            'Passes to Penalty Area': 'opp_pen_pass',
            'Accurate Crosses': 'acc_cross',
            'Progressive Passes': 'prog_pass'
        },
    "passing_types":
        {
            'Live Passes': 'live_pass',
            'Dead Passes': 'dead_pass',
            'Free Kick Passes': 'fk_pass',
            'Completed Through Balls': 'tb_pass',
            'Switch Pass': 'sw_pass',
            'Crosses': 'cross_pass',
            'Throw Ins': 'throw_in',
            'Corner Kicks': 'ck',
            'Inswinging Corner Kicks': 'ck_in',
            'Outswinging Corner Kicks': 'ck_out',
            'Straight Corner Kicks': 'ck_straight',
            'Offside Passes': 'offside_pass',
            'Blocked Passes': 'blocked_pass'
        },
    "gca":
        {
            'Shot Creation': 'sca',
            'Shot Creation (Live Pass)': 'sca_pass_live',
            'Shot Creation (Dead Pass)': 'sca_pass_dead',
            'Shot Creation (Take-On)': 'sca_take_on',
            'Shot Creation (Shot)': 'sca_shot',
            'Shot Creation (Fouled)': 'sca_fouled',
            'Shot Creation (Defending)': 'sca_def',
            'Goal Creation': 'sca',
            'Goal Creation (Live Pass)': 'gca_pass_live',
            'Goal Creation (Dead Pass)': 'gca_pass_dead',
            'Goal Creation (Take-On)': 'gca_take_on',
            'Goal Creation (Shot)': 'gca_shot',
            'Goal Creation (Fouled)': 'gca_fouled',
            'Goal Creation (Defending)': 'gca_def',
        },
    "Possession":
        {
            'Touches': 'touch',
            'Def Pen Area Touches': 'touch_def_pen',
            'Def 3rd Touches': 'touch_def',
            'Mid 3rd Touches': 'touch_mid',
            'Att 3rd Touches': 'touch_att',
            'Att Pen Area Touches': 'touch_att_pen',
            'Live Ball Touches': 'touch_live',
            'Attempted Take-Ons': 'att_take_on',
            'Completed Take-Ons': 'cmp_take_on',
            'Take-On Success (%)': 'take_on_success',
            'Unsuccessful Take-Ons': 'uns_take_on',
            'Carries': 'carry',
            'Carry Distance': 'carry_dist',
            'Progressive Carry Distance': 'carry_prog_dist',
            'Progressive Carries': 'carry_prog',
            'Carries to Att 3rd': 'carry_att_third',
            'Carries to Att Pen Area': 'carry_opp_pen',
            'Mis-controls': 'miscontrol',
            'Dispossessed': 'disposs',
            'Passes Received': 'received',
            'Progressive Passes Received': 'prog_received'
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
