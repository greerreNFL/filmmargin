import pandas as pd
import numpy
import json
import pathlib

from .DataLoader import DataLoader
from .Regression import Regression

import os

## for saving json, need to convert numpy dtypes ##
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def run_development_regressions(total_rounds=1000):
    '''
    Cycles through different variables and calcs efficacy for the model
    '''
    records = []
    ## env vars
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    table = os.getenv('SUPABASE_TABLE')
    ## fields to test ##
    fields = [
        ## overall
        'overall_grade',
        ## offense ##
        'offense_grade', 'pass_grade',
        'pass_block_grade', 'pass_route_grade',
        'run_grade', 'run_block_grade',
        ## def ##
        'defense_grade', 'coverage_defense_grade',
        'pass_rush_defense_grade', 'pass_rush_defense_grade',
        'run_defense_grade',
        ## select opponent ##
        'opponent_overall_grade', 'opponent_offense_grade',
        'opponent_pass_grade', 'opponent_coverage_defense_grade'
    ]
    print('     Loading data...')
    data = DataLoader(url, key, table)
    print('     Running {0} rounds of regressions over {1} feilds...'.format(
        total_rounds, len(fields)
    ))
    for round_num in range(0,total_rounds):
        if round(round_num/50,0) == round_num/50:
            print('          On round {0}'.format(round_num))
        ## init the regression ##
        ## do this upfront so each iterations within the round receives
        ## the same train/test split, which happens on init ##
        reg = Regression(
            df=data.flat_game_grades,
            fields=fields,
            dependent='margin',
            windowing_fields=['season', 'team']
        )
        ## prediction type ##
        for prediction_type in ['margin', 'seasonal_margin']:
            ## update the reg ##
            reg.dependent = prediction_type
            ## cycle through fields ##
            for index, field in enumerate(fields):
                ## drop from set to see impact ##
                fields_ = fields.copy()
                fields_.pop(index)
                reg.fields = fields_
                ## train, score, and append results ##
                reg.train()
                reg.score()
                results = reg.results.copy()
                ## create meta data about the run ##
                meta = {
                    'run_group' : round_num + 1,
                    'dependent' : prediction_type,
                    'field_left_out' : field
                }
                records.append(meta | results)
    ## summarize ##
    print('     Summarizing results')
    df = pd.DataFrame(records)
    ## calc rsq for round ##
    for rsq in ['train_rsq', 'test_rsq']:
        df['group_{0}'.format(rsq)] = df.groupby(
            ['run_group', 'dependent']
        )[rsq].transform('mean')
        ## calc rsq vs group ##
        df['{0}_lift'.format(rsq)] = df[rsq] - df['group_{0}'.format(rsq)]
    ## average ##
    agg = df.groupby(['field_left_out', 'dependent']).agg(
        train_avg_rsq = ('train_rsq', 'mean'),
        test_avg_rsq = ('test_rsq', 'mean'),
        train_lift_when_excluded = ('train_rsq_lift', 'mean'),
        test_lift_when_excluded = ('test_rsq_lift', 'mean'),
    ).reset_index()
    ## sort ##
    agg = agg.sort_values(
        by=['train_lift_when_excluded'],
        ascending=[True]
    ).reset_index(drop=True)
    ## return ##
    return agg


def update_model():
    '''
    Trains a new model on all available data and update the
    config file
    '''
    ## structure for saving to config ##
    output_dict={
        'updated_through' : {
            'season' : None,
            'week' : None
        },
        'descriptive' : {},
        'predictive' : {}
    }
    ## DB config ##
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    table = os.getenv('SUPABASE_TABLE')
    ## fields to use in each model ##
    descriptive_fields=[
        'overall_grade', 'opponent_overall_grade'
    ]
    predictive_fields=[
        'overall_grade', 'opponent_overall_grade',
        'pass_grade', 'run_defense_grade'
    ]
    ## load data ##
    print('Updating the config file with new models')
    print('     Loading data...')
    data = DataLoader(url, key, table)
    ## update config with training week/season data
    output_dict['updated_through']['season'] = data.flat_game_grades['season'].max()
    output_dict['updated_through']['week'] = data.flat_game_grades[
        data.flat_game_grades['season'] == output_dict['updated_through']['season']
    ]['week'].max()
    ## create a reg for each
    print('     Running model...')
    reg_descriptive = Regression(
        df=data.flat_game_grades,
        fields=descriptive_fields,
        dependent='margin',
        windowing_fields=['season', 'team'],
        full_train=True
    )
    reg_predictive = Regression(
        df=data.flat_game_grades,
        fields=predictive_fields,
        dependent='seasonal_margin',
        windowing_fields=['season', 'team'],
        full_train=True
    )
    ## fit ##
    reg_descriptive.train()
    reg_predictive.train()
    ## update struc ##
    ## coefs ##
    ## desc ##
    for index, field in enumerate(descriptive_fields):
        output_dict['descriptive'][field]=reg_descriptive.coefs[index]
    ## pred ##
    for index, field in enumerate(predictive_fields):
        output_dict['predictive'][field]=reg_predictive.coefs[index]
    ## constants ##
    output_dict['descriptive']['intercept'] = reg_descriptive.const
    output_dict['predictive']['intercept'] = reg_predictive.const
    ## save ##
    package_dir = pathlib.Path(__file__).parent.parent.resolve()
    with open('{0}/config.json'.format(package_dir), 'w') as fp:
        json.dump(output_dict, fp, indent=4, cls=NpEncoder)

