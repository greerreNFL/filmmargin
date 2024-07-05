import pandas as pd
import numpy
import pathlib

import nfelodcm

from dotenv import load_dotenv
from ..Supabase import SupabaseClient

## set top level pacakage directory ##
package_dir = pathlib.Path(__file__).parent.parent.parent.resolve()

## load the dot env ##
env_path = '{0}/.env'.format(package_dir)
load_dotenv(env_path)

class DataLoader():
    '''
    Loads all data necessary for package
    '''
    def __init__(self, url, key, table):
        self.db = nfelodcm.load(['games'])
        self.games = self.db['games']
        self.sb_client = SupabaseClient(url, key, table)
        self.game_grades = self.sb_client.get_data()
        self.flat_game_grades = None
        self.add_game_id()
        self.flatten_game_grades()
        self.add_seasonal_margin()

    def add_game_id(self):
        '''
        Replaces game grade ids with nflfastr ids
        '''
        ## drop ids ##
        self.game_grades = self.game_grades.drop(columns=['game_id'])
        ## merge #
        self.game_grades = pd.merge(
            self.game_grades,
            self.games[
                ~pd.isnull(self.games['result'])
            ][[
                'game_id', 'season', 'week',
                'home_team', 'away_team'
            ]].copy(),
            on=['season', 'week', 'home_team', 'away_team'],
            how='left'
        )
    
    def flatten_game_grades(self):
        '''
        Flattens the game grades file by team and week ##
        '''
        self.flat_game_grades = pd.concat([
            self.game_grades[[
                'game_id', 'season', 'week',
                'home_team', 'away_team', 'home_score', 'away_score',
                ## home grades ##
                'home_overall_grade', 'home_offense_grade',
                'home_pass_grade', 'home_pass_block_grade',
                'home_pass_route_grade', 'home_run_grade',
                'home_run_block_grade', 'home_defense_grade',
                'home_coverage_defense_grade', 'home_pass_rush_defense_grade',
                'home_run_defense_grade', 'home_tackle_grade',
                'home_misc_st_grade',
                ## away grades ##
                'away_overall_grade', 'away_offense_grade',
                'away_pass_grade', 'away_pass_block_grade',
                'away_pass_route_grade', 'away_run_grade',
                'away_run_block_grade', 'away_defense_grade',
                'away_coverage_defense_grade', 'away_pass_rush_defense_grade',
                'away_run_defense_grade', 'away_tackle_grade',
                'away_misc_st_grade'
            ]].rename(columns={
                'home_team' : 'team',
                'away_team' : 'opponent',
                'home_score' : 'pf',
                'away_score' : 'pa',
                'home_overall_grade' : 'overall_grade',
                'home_offense_grade' : 'offense_grade',
                'home_pass_grade' : 'pass_grade',
                'home_pass_block_grade' : 'pass_block_grade',
                'home_pass_route_grade' : 'pass_route_grade',
                'home_run_grade' : 'run_grade',
                'home_run_block_grade' : 'run_block_grade',
                'home_defense_grade' : 'defense_grade',
                'home_coverage_defense_grade' : 'coverage_defense_grade',
                'home_pass_rush_defense_grade' : 'pass_rush_defense_grade',
                'home_run_defense_grade' : 'run_defense_grade',
                'home_tackle_grade' : 'tackle_grade',
                'home_misc_st_grade' : 'misc_st_grade',
                'away_overall_grade' : 'opponent_overall_grade',
                'away_offense_grade' : 'opponent_offense_grade',
                'away_pass_grade' : 'opponent_pass_grade',
                'away_pass_block_grade' : 'opponent_pass_block_grade',
                'away_pass_route_grade' : 'opponent_pass_route_grade',
                'away_run_grade' : 'opponent_run_grade',
                'away_run_block_grade' : 'opponent_run_block_grade',
                'away_defense_grade' : 'opponent_defense_grade',
                'away_coverage_defense_grade' : 'opponent_coverage_defense_grade',
                'away_pass_rush_defense_grade' : 'opponent_pass_rush_defense_grade',
                'away_run_defense_grade' : 'opponent_run_defense_grade',
                'away_tackle_grade' : 'opponent_tackle_grade',
                'away_misc_st_grade' : 'opponent_misc_st_grade',
            }),
            self.game_grades[[
                'game_id', 'season', 'week',
                'away_team', 'home_team', 'away_score', 'home_score',
                ## away grades ##
                'away_overall_grade', 'away_offense_grade',
                'away_pass_grade', 'away_pass_block_grade',
                'away_pass_route_grade', 'away_run_grade',
                'away_run_block_grade', 'away_defense_grade',
                'away_coverage_defense_grade', 'away_pass_rush_defense_grade',
                'away_run_defense_grade', 'away_tackle_grade',
                'away_misc_st_grade',
                ## home grades ##
                'home_overall_grade', 'home_offense_grade',
                'home_pass_grade', 'home_pass_block_grade',
                'home_pass_route_grade', 'home_run_grade',
                'home_run_block_grade', 'home_defense_grade',
                'home_coverage_defense_grade', 'home_pass_rush_defense_grade',
                'home_run_defense_grade', 'home_tackle_grade',
                'home_misc_st_grade'
            ]].rename(columns={
                'home_team' : 'opponent',
                'away_team' : 'team',
                'home_score' : 'pa',
                'away_score' : 'pf',
                'home_overall_grade' : 'opponent_overall_grade',
                'home_offense_grade' : 'opponent_offense_grade',
                'home_pass_grade' : 'opponent_pass_grade',
                'home_pass_block_grade' : 'opponent_pass_block_grade',
                'home_pass_route_grade' : 'opponent_pass_route_grade',
                'home_run_grade' : 'opponent_run_grade',
                'home_run_block_grade' : 'opponent_run_block_grade',
                'home_defense_grade' : 'opponent_defense_grade',
                'home_coverage_defense_grade' : 'opponent_coverage_defense_grade',
                'home_pass_rush_defense_grade' : 'opponent_pass_rush_defense_grade',
                'home_run_defense_grade' : 'opponent_run_defense_grade',
                'home_tackle_grade' : 'opponent_tackle_grade',
                'home_misc_st_grade' : 'opponent_misc_st_grade',
                'away_overall_grade' : 'overall_grade',
                'away_offense_grade' : 'offense_grade',
                'away_pass_grade' : 'pass_grade',
                'away_pass_block_grade' : 'pass_block_grade',
                'away_pass_route_grade' : 'pass_route_grade',
                'away_run_grade' : 'run_grade',
                'away_run_block_grade' : 'run_block_grade',
                'away_defense_grade' : 'defense_grade',
                'away_coverage_defense_grade' : 'coverage_defense_grade',
                'away_pass_rush_defense_grade' : 'pass_rush_defense_grade',
                'away_run_defense_grade' : 'run_defense_grade',
                'away_tackle_grade' : 'tackle_grade',
                'away_misc_st_grade' : 'misc_st_grade',
            })
        ])
        self.flat_game_grades.sort_values(
            by=['team', 'season', 'week'],
            ascending=[True, True, True]
        ).reset_index(drop=True)

    def add_seasonal_margin(self):
        '''
        Adds the teams margin across all other games in the season
        '''
        self.flat_game_grades['margin'] = self.flat_game_grades['pf'] - self.flat_game_grades['pa']
        self.flat_game_grades['seasonal_margin'] = self.flat_game_grades.groupby([
            'team', 'season'
        ])['margin'].transform('mean')


