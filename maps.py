from sqlalchemy import Numeric

season_map = {
    "2022-23": 23,
    "2021-22": 22,
    "2020-21": 21,
    "2019-20": 20,
    "2018-19": 19,
    "2017-18": 18}

comp_map = {
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

position_list = [
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

# the array signifies a numerator, denominator, factor and output data type
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
    "launch_perc": ["att_launch_non_goal_kick", "att_pass_non_goal_kick", 100, Numeric(4, 1)],
}

combined = {
    "non_penalty_goals_and_xa": ["non_penalty_goals", "xa", 'shooting', 'passing'],
    "npxg_and_xa": ["npxg", "xa", 'shooting', 'passing'],
    "non_penalty_goals_and_assists": ["non_penalty_goals", "assist", 'shooting', 'passing'],
}

axis_options = {
    "combined":
        {
            "Non Penalty Goals + xA": 'non_penalty_goals_and_xa',
            "npxG + xA": "npxg_and_xa",
            "Non Penalty Goals + Assists": 'non_penalty_goals_and_assists'
        },
    "shooting":
        {
            "Non Penalty Goals": 'non_penalty_goals',
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
    "possession":
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
    "defense":
        {
            'Attempted Tackles': 'att_tackle',
            'Completed Tackles': 'cmp_tackle',
            'Attempted Tackles (Def 3rd)': 'att_tackle_def',
            'Attempted Tackles (Mid 3rd)': 'att_tackle_mid',
            'Attempted Tackles (Att 3rd)': 'att_tackle_att',
            'Attempted Dribbler Tackles': 'att_drib_tackle',
            'Completed Dribbler Tackles': 'cmp_drib_tackle',
            'Dribbler Tackle Success (%)': 'dribbler_tackle_perc',
            'Unsuccessful Dribbler Tackles': 'uns_drib_tackle',
            'Blocks': 'block',
            'Blocked Shots': 'block_shot',
            'Blocked Passes': 'block_pass',
            'Interceptions': 'intercept',
            'Tackles + Interceptions': 'tackle_plus_intercept',
            'Clearances': 'clearance',
            'Errors Leading to Shot': 'error'
        },
    "misc":
        {
            'Yellow Cards': 'y_card',
            'Red Cards': 'r_card',
            'Two Yellow Cards': 'two_y_card',
            'Fouls': 'fouls',
            'Fouled': 'fouled',
            'Offsides': 'offside',
            'Penalties Won': 'pens_won',
            'Penalties Conceded': 'pens_con',
            'Own Goals': 'own_goal',
            'Posession Recovered': 'recov',
            'Aerials Won': 'aerials_won',
            'Aerials Lost': 'aerials_lost',
            'Aerials Attempted': 'aerials_attempted',
            'Aerial Success (%)': 'aerial_success'
        },
    "playingtime":
        {
            'Matches': 'matches',
            'Minutes': 'minutes',
            'Nineties': 'nineties',
            'Starts': 'starts',
            'Matches Completed': 'completed',
            'Substitute Appearances': 'sub',
            'Unused Substitute': 'sub_unused',
            'Team Goals While on Pitch': 'onpitch_goals',
            'Team Goals Against While on Pitch': 'onpitch_goals_ag',
            'Team Goals +/- While on Pitch': 'onpitch_goals_delta',
            'Team xG While on Pitch': 'onpitch_xg',
            'Team xGA While on Pitch': 'onpitch_xga',
            'Team xG +/- While on Pitch': 'onpitch_xg_delta',
        },
    "keepers":
        {
            'Matches as Goalkeeper': 'matches_gk',
            'Starts as Goalkeeper':  'starts_gk',
            'Minutes as Goalkeeper': 'minutes_gk',
            'Nineties as Goalkeeper': 'nineties_gk',
            'Goals Against': 'goals_ag',
            'Shots on Target Against': 'sot_ag',
            'Saves': 'saves',
            'Save Percentage (%)': 'save_perc',
            'Wins as Goalkeeper': 'gk_won',
            'Draws as Goalkeeper': 'gk_drew',
            'Losses as Goalkeeper': 'gk_lost',
            'Clean Sheets': 'cs',
            'Clean Sheet Percentage (%)': 'clean_sheet_perc',
            'Penalties Faced': 'pk_att_ag',
            'Penalties Scored Against': 'pk_scored_ag',
            'Penalties Saved': 'pk_saved',
            'Penalty Save Percentage (%)': 'pk_save_perc',
            'Penalties Missed Against': 'pk_missed_ag',
        },
    "keepersadv":
        {
            'Free Kicks Against': 'fk_ag',
            'Corner Kicks Against': 'ck_ag',
            'Own Goals Against': 'og_ag',
            'Post Shot xG Faced': 'ps_xg',
            'Post Shot xG +/-': 'ps_xg_delta',
            'Attempted Launches': 'att_launch',
            'Completed Launches': 'cmp_launch',
            'Launch Accuracy (%)': 'launch_completion',
            'Passes Launched (%)': 'launch_perc',
            'Attempted Keeper Passes': 'att_gk_pass',
            'Attempted Keeper Passes (exc. dead)': 'att_pass_non_goal_kick',
            'Attempted Throws': 'att_gk_throw',
            'Avg Keeper Pass Length (exc. dead)': 'avg_gk_pass_len',
            'Goal Kicks': 'att_goal_kick',
            'Launched Goal Kicks': 'goal_kicks_launched',
            'Goal Kick Launches (%)': 'goal_kick_launch_perc',
            'Short Goal Kicks': 'att_launch_non_goal_kick',
            'Avg Goal Kick Length': 'avg_goal_kick_len',
            'Crosses Faced': 'att_cross_ag',
            'Crosses Stopped': 'stop_cross_ag',
            'Crosses Stopped (%)': 'cross_stop_perc',
            'Defensive Actions Outside Pen': 'def_outside_pen',
            'Average Distance of Defensive Actions': 'avg_def_outside_pen_dist',
            'Goal Kicks Launched': 'goal_kicks_launched'
        }
}

intial_label_data = [{'x': 0.344, 'y': 0.312, 'name': 'Robben'},
                     {'x': 0.32, 'y': 0.382, 'name': 'Müller'},
                     {'x': 0.56, 'y': 0.361, 'name': 'Neymar'},
                     {'x': 0.596, 'y': 0.21, 'name': 'Salah'},
                     {'x': 0.339, 'y': 0.386, 'name': 'María'},
                     {'x': 0.73, 'y': 0.168, 'name': 'Agüero'},
                     {'x': 0.196, 'y': 0.35, 'name': 'Payet'},
                     {'x': 1.031, 'y': 0.133, 'name': 'Haaland'},
                     {'x': 0.291, 'y': 0.373, 'name': 'Bruyne'},
                     {'x': 0.307, 'y': 0.334, 'name': 'Dembélé'},
                     {'x': 0.791, 'y': 0.285, 'name': 'Mbappé'},
                     {'x': 0.476, 'y': 0.32, 'name': 'Iličić'},
                     {'x': 0.408, 'y': 0.292, 'name': 'Sané'},
                     {'x': 0.873, 'y': 0.163, 'name': 'Lewandowski'},
                     {'x': 0.352, 'y': 0.346, 'name': 'Coman'},
                     {'x': 0.676, 'y': 0.132, 'name': 'Núñez'},
                     {'x': 0.714, 'y': 0.425, 'name': 'Messi'},
                     {'x': 0.441, 'y': 0.269, 'name': 'Nkunku'},
                     {'x': 0.056, 'y': 0.377, 'name': 'Raum'},
                     {'x': 0.323, 'y': 0.343, 'name': 'Rodríguez'},
                     {'x': 0.116, 'y': 0.337, 'name': 'Kimmich'},
                     {'x': 0.541, 'y': 0.348, 'name': 'Gnabry'},
                     {'x': 0.697, 'y': 0.125, 'name': 'Bakambu'},
                     {'x': 0.574, 'y': 0.212, 'name': 'Muriel'},
                     {'x': 0.388, 'y': 0.278, 'name': 'Mahrez'}]