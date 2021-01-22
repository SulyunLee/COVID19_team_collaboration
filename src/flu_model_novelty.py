import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot  as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale

def correlation_analysis(df, feature_colnames, target_colname):
    for feature in feature_colnames:
        corr = df[feature].corr(df[target_colname])
        # corr, p_val = pearsonr(df[feature], df[target_colname])
        # print("Corr({}, {}): {} ({})".format(feature, target_colname, corr, p_val))
        print("Corr({}, {}): {}".format(feature, target_colname, corr))
    
def calculate_vif(X, thresh=10.0):
    print(thresh)
    dropped = True
    while dropped:
        variables = X.columns
        dropped = False
        print(X.columns)
        vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
        print(vif)
        max_vif = max(vif)
        print(max(vif))
        if max_vif > thresh:
            maxloc = vif.index(max_vif)
            print(X.columns.tolist()[maxloc])
            X = X.drop([X.columns.tolist()[maxloc]], axis=1)
            dropped = True

    return X

def linreg_model(df, feature_colnames, target_colname, write_filename):
    '''
    Input:
      - df: dataframe that contains the instances for linear regression modeling which
            includes features and target variables
      - feature_colnames (list): the list of column names to be used for features
      - target_colname (str): the string of the target variable column name

    Output:
    '''
    X = df[feature_colnames]
    aftervif = calculate_vif(X)
    formula = "{} ~ {}".format(target_colname, "+".join(aftervif.columns))


    model = smf.ols(formula=formula, data=df).fit()
    with open("../results/{}.csv".format(write_filename), "w") as fh:
        fh.write(model.summary().as_csv())

    return model

if __name__ == "__main__":
    print("Loading modeling data...")

    flu_papers_modeling_filename = "../dataset/FLU_papers_modeling_with_features.csv"
    flu_paper_df = pd.read_csv(flu_papers_modeling_filename)

    print("Cleaning data...")
    cultural_sim_replaced = flu_paper_df['cultural_similarity'].replace(-1,np.nan)
    flu_paper_df = flu_paper_df.assign(cultural_similarity = cultural_sim_replaced)

    team_size_log_transformed = np.log(flu_paper_df['team_size']+1)
    max_hindex_log_transformed = np.log(flu_paper_df['max_hindex']+1)

    flu_paper_df = flu_paper_df.assign(team_size_log = team_size_log_transformed)
    flu_paper_df = flu_paper_df.assign(max_hindex_log = max_hindex_log_transformed)

    # define variables
    feature_var = ["avg_tdsim", "new_tie_rate", "hindex_gini","cultural_similarity", "topic_familiarity", "max_hindex_log", "time_since_pub", "team_size_log", "prac_affil_rate"] + ["topic_distr{}".format(i) for i in range(1,20)]

    # drop na rows
    flu_paper_df = flu_paper_df.dropna(subset=feature_var+["novelty_10perc", "novelty_5perc", "novelty_1perc"])
    print("Number of instances: {}".format(flu_paper_df.shape[0]))

    # plot novelty variable
    sns.distplot(flu_paper_df['novelty_10perc'])
    plt.title("Distribution of novelty (10%)")
    plt.savefig("../plots/flu_novelty_10perc.png")
    plt.close()

    sns.distplot(flu_paper_df['novelty_5perc'])
    plt.title("Distribution of novelty (5%)")
    plt.savefig("../plots/flu_novelty_5perc.png")
    plt.close()

    sns.distplot(flu_paper_df['novelty_1perc'])
    plt.title("Distribution of novelty (1%)")
    plt.savefig("../plots/flu_novelty_1perc.png")
    plt.close()
    # Standardize
    X = np.array(flu_paper_df[feature_var])
    X_scaled = scale(X)

    flu_paper_scaled_df = flu_paper_df.copy()
    flu_paper_scaled_df[feature_var] = X_scaled

    print("Modeling..")
    novelty_10perc_model = linreg_model(flu_paper_scaled_df, feature_var, "novelty_10perc", "flu_linreg_result_novelty_10perc")
    novelty_5perc_model = linreg_model(flu_paper_scaled_df, feature_var, "novelty_5perc", "flu_linreg_result_novelty_5perc")
    novelty_1perc_model = linreg_model(flu_paper_scaled_df, feature_var, "novelty_1perc", "flu_linreg_result_novelty_1perc")

    ## Model 0: control variable only
    control_var = ["max_hindex_log", "time_since_pub", "team_size_log", "prac_affil_rate"] + ["topic_distr{}".format(i) for i in range(1,20)]
    control_novelty_10perc_model = linreg_model(flu_paper_scaled_df, control_var, "novelty_10perc", "flu_linreg_control_novelty_10perc")
    control_novelty_5perc_model = linreg_model(flu_paper_scaled_df, control_var, "novelty_5perc", "flu_linreg_control_novelty_5perc")
    control_novelty_1perc_model = linreg_model(flu_paper_scaled_df, control_var, "novelty_1perc", "flu_linreg_control_novelty_1perc")



