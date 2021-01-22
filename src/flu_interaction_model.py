
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot  as plt
from scipy.stats import pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import scale

def calculate_vif(X, thresh=10.0):
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

if __name__ == "__main__":
    print("Loading modeling data...")

    flu_papers_modeling_filename = "../dataset/FLU_papers_modeling_with_features.csv"
    flu_paper_df = pd.read_csv(flu_papers_modeling_filename)

    print("Cleaning data...")
    cultural_sim_replaced = flu_paper_df['cultural_similarity'].replace(-1,np.nan)
    flu_paper_df = flu_paper_df.assign(cultural_similarity = cultural_sim_replaced)

    team_size_log_transformed = np.log(flu_paper_df['team_size']+1)
    max_hindex_log_transformed = np.log(flu_paper_df['max_hindex']+1)
    citation_count_log_transformed = np.log(flu_paper_df['citation_count']+1)

    flu_paper_df = flu_paper_df.assign(team_size_log = team_size_log_transformed)
    flu_paper_df = flu_paper_df.assign(max_hindex_log = max_hindex_log_transformed)
    flu_paper_df = flu_paper_df.assign(citation_count_log = citation_count_log_transformed)

    # define variables
    feature_var = ["avg_tdsim", "new_tie_rate", "hindex_gini","cultural_similarity", "topic_familiarity", "max_hindex_log", "time_since_pub", "team_size_log", "prac_affil_rate"] + ["topic_distr{}".format(i) for i in range(1,20)]
    target_colname = "citation_count_log"

    # drop na rows
    flu_paper_df = flu_paper_df.dropna(subset=feature_var)
    print("Number of instances: {}".format(flu_paper_df.shape[0]))

    # Standardize
    X = np.array(flu_paper_df[feature_var])
    X_scaled = scale(X)

    flu_paper_scaled_df = flu_paper_df.copy()
    flu_paper_scaled_df[feature_var] = X_scaled

    X = flu_paper_scaled_df[feature_var]
    aftervif = calculate_vif(X)

    ##### Model 2: Square of new_tie_rate added
    formula = "{} ~ I(new_tie_rate**2) + {}".format(target_colname, "+".join(aftervif.columns))
    square_model = smf.ols(formula=formula, data=flu_paper_scaled_df).fit()

    # write_filename = "flu_linreg_curvilinear_new_tie_rate_citation"
    # with open("../results/{}.csv".format(write_filename), "w") as fh:
        # fh.write(square_model.summary().as_csv())

    ##### Model 3: Interaction of new_tie_rate and topic_familiarity added
    formula = "{} ~ new_tie_rate:topic_familiarity + {}".format(target_colname, "+".join(aftervif.columns))
    interaction_model = smf.ols(formula=formula, data=flu_paper_scaled_df).fit()

    write_filename = "flu_linreg_interaction_new_tie_rate_topic_familiarity"
    with open("../results/{}.csv".format(write_filename), "w") as fh:
        fh.write(interaction_model.summary().as_csv())

