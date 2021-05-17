import numpy as np
import pandas as pd

class test():
    '''
    Initialize an object that can later be fit to an array of predicted probabilities
    and an array of observed targets to test the performance of a binary classifier.
    The fitted object calculates the Kolmogorov-Smirnov test, the odds between observations
    of class 0 and observations of class 1 and a results table with other metrics.

    Attributes
    ----------
    obs:
        The number of observations in the data.
    class0: int
        The number of observations labeled 0 in `y_true`.
    class1: int
        The number of observations labeled 1 in `y_true`.
    ks: float
        The resulting coefficient of the Kolmogorov-Smirnoff test.
    table: pandas.DataFrame
        Results table of the test. The columns are:
        - bin
            Consecutive integers for the probability intervals in ascending order.
        - range
            Lower and upper limit of the predicted probabilites.
        - size
            Number of observations contained within each interval.
        - class0
            Number of observations of class 0 contained within each interval.
        - class1
            Number of observations of class 1 contained within each interval.
        - odds
            The ratio of `class0` divided by `class1` within each interval.
        - class0_rate
            The ratio of observations labeled 0 within each interval.
        - class1_rate
            The ratio of observations labeled 1 within each interval.
        - remainder_total
            The number of observations that have a probability greater than
            the lower limit of each interval.
        - remainder_class0
            The number of observations of class 0 that have a predicted probability
            greater than the lower limit of each interval.
        - remainder_class1
            The number of observations of class 1 that have a predicted probability
            greater than the lower limit of each interval.
        - cumulative_class0
            The cumulative frequency for each interval of observations of class 0.
        - cumulative_class1
            The cumulative frequency for each interval of observations of class 1.
        - abs_difference
            The absolute difference between the cumulative ratio of both classes.
    
    Methods
    -------
    fit(y_proba, y_true, bins, round_range)

    '''
    def fit(self, y_proba, y_true, bins=10, round_range=False):
        '''
        Fit the initialized object to the observed labels.

        Parameters
        ----------
        y_proba: array-like
            Array-like structure that has the predicted probabilities for each observation.
        y_true: array-like
            Array-like structure that has the observed classes for each observation.
        '''
        n0, n1 = len(y_proba) - y_true.sum(), y_true.sum()
        t = pd.DataFrame({'y_proba':y_proba, 'y_true':y_true})
        t['range'] = pd.qcut(x=y_proba, q=bins, labels=None, retbins=False, duplicates='drop')
        t = t.groupby('range')['y_true'].agg(['size','sum']).reset_index()
        t.columns = ['range','size','class1']
        t['bin'] = range(1, len(t) + 1)
        t['class0'] = t['size'] - t['class1']
        t['odds'] = t['class0'].div(t['class1'])
        t['class0_rate'] = t['class0'].div(t['size'])
        t['class1_rate'] = t['class1'].div(t['size'])
        t['remainder_total'] = t['size'].sum() - t['size'].shift(fill_value=0).cumsum()
        t['remainder_class0'] = t['class0'].sum() - t['class0'].shift(fill_value=0).cumsum()
        t['remainder_class1'] = t['class1'].sum() - t['class1'].shift(fill_value=0).cumsum()
        t['cumulative_class0'] = t['class0'].cumsum()
        t['cumulative_class1'] = t['class1'].cumsum()
        t['abs_difference'] = (t['cumulative_class0'].div(n0) - t['cumulative_class1'].div(n1)).abs()
        if round_range:
            t['range'].apply(lambda x: pd.Interval(int(round(x.left)), int(round(x.right))), axis=1)
        t = t[['bin','range','size','class0','class1','odds','class0_rate',
               'class1_rate','remainder_total','remainder_class0','remainder_class1',
               'cumulative_class0','cumulative_class1','abs_difference']]
        self.obs = int(n0 + n1)
        self.class0 = int(n0)
        self.class1 = int(n1)
        self.ks = t['abs_difference'].max()
        self.table = t
