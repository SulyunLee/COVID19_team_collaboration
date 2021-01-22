
'''
This script compares the coefficients obtained from regression model
in two different models.
Comparison: Flu model coefficient vs. COVID19 model coefficient
Reference: https://stats.stackexchange.com/questions/93540/testing-equality-of-coefficients-from-two-different-regressions
'''

from numpy import sqrt, abs, round
from scipy.stats import norm
import argparse
import pandas as pd


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-flu_std', '--flu_std', type=float)
    # parser.add_argument('-covid_std', '--covid_std', type=float)
    # parser.add_argument('-flu_coef', '--flu_coef', type=float)
    # parser.add_argument('-covid_coef', '--covid_coef', type=float)

    # args = parser.parse_args()
    # flu_std = args.flu_std
    # covid_std = args.covid_std
    # flu_coef = args.flu_coef
    # covid_coef = args.covid_coef

    input_filename = "../results/coef_difference_test.csv"
    input_df = pd.read_csv(input_filename)

    for idx, row in input_df.iterrows():
        variable = row.variable
        covid_std = row.covid_std
        covid_coef = row.covid_coef
        flu_std = row.flu_std
        flu_coef = row.flu_coef

        coef_difference = covid_coef - flu_coef
        derivative = sqrt(covid_std**2 + flu_std**2)

        z_score = coef_difference / derivative
        p_val = 2*(1-norm.cdf(abs(z_score)))

        print("Test for {}".format(variable))
        print("COVID19 coef: {}, flu coef: {}, Difference: {}, p-value:{}".format(covid_coef, flu_coef, coef_difference, p_val))





