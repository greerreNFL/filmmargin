import pandas as pd
import numpy
import statsmodels.api as sm


class Regression():
    '''
    Runs regressions to measure the predictiveness
    Using full_train=Train, will use the entire DF
    to train, which should be used when updating the config
    '''
    def __init__(
        self, df, fields, dependent, windowing_fields, full_train=False
    ):
        self.df = df
        self.fields = fields
        self.dependent = dependent
        self.windowing_fields = windowing_fields
        self.train_df, self.test_df = self.window()
        self.full_train=full_train
        ## trained vars ##
        self.coefs = []
        self.const = 0
        ## results ##
        self.results = {}
    
    def window(self):
        '''
        Splits the data into a training and test set
        '''
        ## create a df of all unique combos ##
        temp = self.df[self.windowing_fields].groupby(self.windowing_fields).head(1)
        ## randomly assign training and test ##
        mask = numpy.random.rand(len(temp)) < 0.6
        train = temp[mask].copy()
        train['sample'] = 'train'
        test = temp[~mask].copy()
        test['sample'] = 'test'
        ## combine ##
        temp = pd.concat([
            train, test
        ])
        ## add back to df ##
        temp = pd.merge(
            self.df,
            temp,
            on=self.windowing_fields,
            how='left'
        )
        ## return ##
        return (
            temp[temp['sample']=='train'].reset_index(drop=True),
            temp[temp['sample']=='test'].reset_index(drop=True)
        )
    
    def calc_rsq(self, df):
        '''
        Returns an rmse
        '''
        rss = numpy.sum((df[self.dependent] - df['prediction']) ** 2)
        tss = numpy.sum((df[self.dependent] - df[self.dependent].mean()) ** 2)
        return 1 - min(1, rss/tss)

    def train(self):
        '''
        Trains the regression of a windowed test set
        '''
        ## drop records with NAs in the frame ##
        temp = self.train_df.copy()
        temp = temp.dropna(subset=self.fields)
        if len(temp) != len(self.train_df):
            print('     Warning - some fields contained NAs. {0} records removed.'.format(
                len(self.train_df) - len(temp)
            ))
            print('          Fields: {0}'.format(', '.join(self.fields)))
        ## add constant ##
        temp['intercept'] = 1
        ## train ##
        model = sm.OLS(
            temp[self.dependent],
            temp[self.fields + ['intercept']],
            hasconst=True
        ).fit()
        ## clear coefs ##
        self.coefs = []
        ## update trained variables ##
        for index, var in enumerate(model.params.tolist()):
            if index == len(model.params) - 1:
                self.const = var
            else:
                self.coefs.append(var)
    
    def apply_prediction(self, df):
        '''
        Applies prediction to a df from the trained model
        '''
        df['prediction'] = self.const
        for index, coef in enumerate(self.coefs):
            df['prediction'] = df['prediction'] + (df[self.fields[index]] * coef)
        ## return the df ##
        return df

    def score(self):
        '''
        Scores the testing and training sets
        '''
        ## make predictions ##
        self.train_df = self.apply_prediction(self.train_df)
        self.test_df = self.apply_prediction(self.test_df)
        ## calc rsqs ##
        train_rsq = self.calc_rsq(self.train_df)
        test_rsq = self.calc_rsq(self.test_df)
        ## update record ##
        self.results = {
            'train_rsq' : train_rsq,
            'test_rsq' : test_rsq
        }