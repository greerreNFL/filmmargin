import pandas as pd
import json
import pathlib

from .DataLoader import DataLoader

import os

## suppress logging ##
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

def update_margins():
    '''
    Calculates the film margins
    '''
    print('Updating film margins')
    ## env vars
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    table = os.getenv('SUPABASE_TABLE')
    print('     Loading config...')
    package_dir = pathlib.Path(__file__).parent.parent.resolve()
    with open('{0}/config.json'.format(package_dir), 'r') as fp:
        config = json.load(fp)
    print('     Loading data...')
    data = DataLoader(url, key, table)
    ## calculate ##
    print('     Calcualting margins...')
    ## descriptive ##
    data.flat_game_grades['film_margin'] = config['descriptive']['intercept']
    for k, v in config['descriptive'].items():
        if k != 'intercept':
            data.flat_game_grades['film_margin'] = (
                data.flat_game_grades['film_margin'] +
                (data.flat_game_grades[k] * v)
            )
    ## predictive
    data.flat_game_grades['film_margin_predictive'] = config['predictive']['intercept']
    for k, v in config['predictive'].items():
        if k != 'intercept':
            data.flat_game_grades['film_margin_predictive'] = (
                data.flat_game_grades['film_margin_predictive'] +
                (data.flat_game_grades[k] * v)
            )
    ## create margin ##
    data.flat_game_grades['margin'] = (
        data.flat_game_grades['pf'] -
        data.flat_game_grades['pa']
    )
    data.flat_game_grades['film_margin_old_model'] = (
        -87.728 +
        1.263 * data.flat_game_grades['overall_grade']
    )
    ## save ##
    print('     Saving...')
    data.flat_game_grades[[
        'game_id', 'season', 'week', 'team', 'opponent',
        'pf', 'pa', 'margin', 'film_margin', 'film_margin_predictive',
        'film_margin_old_model'
    ]].to_csv('{0}/film_margins.csv'.format(package_dir))