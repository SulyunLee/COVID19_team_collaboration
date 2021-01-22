
import pandas as pd
import numpy as np
import random
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
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

    # define variables
    feature_var = ["avg_tdsim", "new_tie_rate", "hindex_gini","cultural_similarity", "topic_familiarity", "max_hindex_log", "time_since_pub", "team_size_log", "prac_affil_rate"] + ["topic_distr{}".format(i) for i in range(1,20)]
    target_colname = "citation_count_log"
    # drop na rows
    covid19_paper_df = covid19_paper_df.dropna(subset=feature_var)
    print("Number of instances: {}".format(covid19_paper_df.shape[0]))

    # Standardize
    X = np.array(covid19_paper_df[feature_var])
    X_scaled = scale(X)

    covid19_paper_scaled_df = covid19_paper_df.copy()
    covid19_paper_scaled_df[feature_var] = X_scaled

    X = covid19_paper_scaled_df[feature_var]
    aftervif = calculate_vif(X)

    ##### Model 2: Square of new_tie_rate added
    formula = "{} ~ I(new_tie_rate**2) + {}".format(target_colname, "+".join(aftervif.columns))

    square_model = smf.ols(formula=formula, data=covid19_paper_scaled_df).fit()

    # write_filename = "covid19_linreg_curvilinear_new_tie_rate_citation"
    # with open("../results/{}.csv".format(write_filename), "w") as fh:
        # fh.write(model.summary().as_csv())

    ##### Model 3: Interaction of new_tie_rate and topic_familiarity added
    formula = "{} ~ new_tie_rate:topic_familiarity + {}".format(target_colname, "+".join(aftervif.columns))
    interaction_model = smf.ols(formula=formula, data=covid19_paper_scaled_df).fit()

    write_filename = "covid19_linreg_interaction_new_tie_rate_topic_familiarity"
    with open("../results/{}.csv".format(write_filename), "w") as fh:
        fh.write(interaction_model.summary().as_csv())

    # plot the interaction term
    covid19_paper_scaled_df['new_tie_rate_med'] = covid19_paper_scaled_df.new_tie_rate > covid19_paper_scaled_df.new_tie_rate.median() 
    covid19_paper_scaled_df['new_tie_rate_med'] = np.where(covid19_paper_scaled_df.new_tie_rate_med == False, "Below Median", "Above Median")
    sns.lmplot(x="topic_familiarity", y="citation_count_log", hue="new_tie_rate_med", data=covid19_paper_scaled_df,\
            ci=None, size=5, aspect=2.5, scatter=False)
    # plt.ylim(0, 8)
    plt.savefig("../plots/interaction_term_scatter_plot.png")
    plt.close()


