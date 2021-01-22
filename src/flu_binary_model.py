


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

if __name__ == "__main__":
    print("Loading modeling data...")

    flu_papers_modeling_filename = "../dataset/FLU_papers_modeling_with_features.csv"
    flu_paper_df = pd.read_csv(flu_papers_modeling_filename)

    # replace average topic distr similarity -1 as nan
    # value of -1 means zero or one authors have abstracts available
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
    target_colname = "citation_counts_binary"

    # drop na rows
    flu_paper_df = flu_paper_df.dropna(subset=feature_var + ["citation_count_log"])
    print("Number of instances: {}".format(flu_paper_df.shape[0]))

    # change the citation counts into binary variable
    # Top 20% vs 80% -> cut point: citation_counts_log = up to top 437 papers
    # perc = "20%"
    # cut = flu_paper_df.citation_count_log.sort_values().iloc[-437] # 1.0986
    # Top 10% vs 90% -> cut point: citation_counts_log = up to 277 papers
    # perc = "10%"
    # cut = flu_paper_df.citation_count_log.sort_values().iloc[-218] # 1.3863
    # Top 5% vs 95% -> cut point: citation_counts_log = up to 165 papers
    # perc = "5%"
    # cut = flu_paper_df.citation_count_log.sort_values().iloc[-109] # 1.6094
    # Top 1% vs 99% -> cut point: citation_counts_log = up to 26 papers
    perc = "1%"
    cut = flu_paper_df.citation_count_log.sort_values().iloc[-21] # 2.3026


    citation_counts_binary = np.where(flu_paper_df.citation_count_log >= cut,1,0)
    flu_paper_df = flu_paper_df.assign(citation_counts_binary = citation_counts_binary)
    

    # Standardize
    X = np.array(flu_paper_df[feature_var])
    X_scaled = scale(X)

    flu_paper_scaled_df = flu_paper_df.copy()
    flu_paper_scaled_df[feature_var] = X_scaled
    X = flu_paper_scaled_df[feature_var]

    # remove multi-collinearity
    aftervif = calculate_vif(X)

    print("Modeling...")
    # formula = "{} ~ {}".format(target_colname, "+".join(aftervif.columns))
    model = sm.OLS(flu_paper_scaled_df[target_colname], flu_paper_scaled_df[aftervif.columns]).fit()

    write_filename = "flu_logit_model"
    with open("../results/{}_top{}.csv".format(write_filename, perc), "w") as fh:
        fh.write(model.summary().as_csv())



