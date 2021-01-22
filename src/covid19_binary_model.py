

import pandas as pd
import numpy as np
import random
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import scale

def calculate_vif(X, thresh=10.0):
    print(thresh)
    dropped = True
    while dropped:
        variables = X.columns
        dropped = False
        # print(X.columns)
        vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
        # print(vif)
        max_vif = max(vif)
        # print(max(vif))
        if max_vif > thresh:
            maxloc = vif.index(max_vif)
            # print(X.columns.tolist()[maxloc])
            X = X.drop([X.columns.tolist()[maxloc]], axis=1)
            dropped = True

    return X

def generate_binary_dv(perc, df):
    cut_point = int(df.shape[0]*perc)
    cut = df.citation_count_log.sort_values().iloc[-cut_point]
    citation_counts_binary = np.where(df.citation_count_log > cut, 1, 0)
    df['citation_counts_binary_{}perc'.format(int(perc*100))] = citation_counts_binary

    return df


def fit_binary_model(perc, covid19_paper_scaled_df, aftervif):

    model = sm.OLS(covid19_paper_scaled_df["citation_counts_binary_{}perc".format(int(perc*100))], covid19_paper_scaled_df[aftervif.columns]).fit()

    write_filename = "covid19_logit_model"
    with open("../results/{}_top{}%.csv".format(write_filename, int(perc*100)), "w") as fh:
        fh.write(model.summary().as_csv())

    return model

if __name__ == "__main__":
    print("Loading modeling data...")
    covid19_papers_modeling_filename = "../dataset/COVID19_papers_modeling_with_features.csv"
    covid19_paper_df = pd.read_csv(covid19_papers_modeling_filename)

    print("Cleaning data...")
    citation_count_replaced = covid19_paper_df['citation_count'].replace(-1,np.nan)
    covid19_paper_df = covid19_paper_df.assign(citation_count = citation_count_replaced)


    # replace average topic distr similarity -1 as nan
    # value of -1 means zero or one authors have abstracts available
    avg_tdsim_replaced = covid19_paper_df['avg_tdsim'].replace(-1, np.nan)
    covid19_paper_df = covid19_paper_df.assign(avg_tdsim = avg_tdsim_replaced)

    team_size_log_transformed = np.log(covid19_paper_df['team_size']+1)
    max_hindex_log_transformed = np.log(covid19_paper_df['max_hindex']+1)
    citation_count_log_transformed = np.log(covid19_paper_df['citation_count']+1)
    mention_log_transformed = np.log(covid19_paper_df['mention']+1)

    covid19_paper_df = covid19_paper_df.assign(team_size_log = team_size_log_transformed)
    covid19_paper_df = covid19_paper_df.assign(max_hindex_log = max_hindex_log_transformed)
    covid19_paper_df = covid19_paper_df.assign(citation_count_log = citation_count_log_transformed)
    covid19_paper_df = covid19_paper_df.assign(mention_log = mention_log_transformed)

    # Generate binary DVs
    covid19_paper_df = generate_binary_dv(0.2, covid19_paper_df)
    covid19_paper_df = generate_binary_dv(0.1, covid19_paper_df)
    covid19_paper_df = generate_binary_dv(0.05, covid19_paper_df)

    # define variables
    feature_var = ["avg_tdsim", "new_tie_rate", "hindex_gini","cultural_similarity", "topic_familiarity", "max_hindex_log", "time_since_pub", "team_size_log", "prac_affil_rate"] + ["topic_distr{}".format(i) for i in range(1,20)]
    target_colname = "citation_counts_binary"

    # drop na rows
    covid19_paper_df = covid19_paper_df.dropna(subset=feature_var + ["citation_count_log"])

    print("Number of instances: {}".format(covid19_paper_df.shape[0]))


    # Standardize
    X = np.array(covid19_paper_df[feature_var])
    X_scaled = scale(X)

    covid19_paper_scaled_df = covid19_paper_df.copy()
    covid19_paper_scaled_df[feature_var] = X_scaled
    X = covid19_paper_scaled_df[feature_var]

    # remove multi-collinearity
    aftervif = calculate_vif(X)

    model_20perc = fit_binary_model(0.2, covid19_paper_scaled_df, aftervif)
    model_10perc = fit_binary_model(0.1, covid19_paper_scaled_df, aftervif)
    model_5perc = fit_binary_model(0.05, covid19_paper_scaled_df, aftervif)


