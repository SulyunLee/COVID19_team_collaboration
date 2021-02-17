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

def linreg_curvilinear_model(df, feature_colnames, target_colname, curvilinear_var, write_filename):
    X = df[feature_colnames]
    aftervif = calculate_vif(X)

    formula = "{} ~ I({}**2) + {}".format(target_colname, curvilinear_var, "+".join(aftervif.columns))
    print(formula)

    model = smf.ols(formula=formula, data=df).fit()

    print(write_filename)
    with open("../results/{}.csv".format(write_filename), "w") as fh:
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

    # Export citation count distribution
    citation_distr = covid19_paper_df.citation_count
    citation_distr.to_csv("../results/covid19_citation_distribution.csv", index=False)

    # define variables
    control_var = ["avg_tdsim", "new_tie_rate", "hindex_gini","cultural_similarity", "topic_familiarity_var", "max_hindex_log", "time_since_pub", "team_size_log", "prac_affil_rate"] + ["topic_distr{}".format(i) for i in range(1,20)]
    predictor_var = "topic_familiarity"
    target_var = "citaion_count_log"

    # drop na rows
    covid19_paper_df = covid19_paper_df.dropna(subset=[predictor_var] + control_var)
    print("Number of instances: {}".format(covid19_paper_df.shape[0]))


    # Standardize
    X = np.array(covid19_paper_df[[predictor_var] + control_var])
    X_scaled = scale(X)

    covid19_paper_scaled_df = covid19_paper_df.copy()
    covid19_paper_scaled_df[[predictor_var] + control_var] = X_scaled

    # correlation matrix
    correlation_matrix = covid19_paper_df[[predictor_var] + control_var + ["citation_count_log", "citation_counts_binary_5perc", "novelty_10perc"]].corr()
    correlation_matrix.to_csv("../results/covid19_modeling_variables_corr_matrix.csv")

    print("Modeling...")
    citation_count_model = linreg_model(covid19_paper_scaled_df, [predictor_var] + control_var, "citation_count_log", "covid19_linreg_result_citation_count")
    model_result = pd.concat([citation_count_model.params, citation_count_model.bse, citation_count_model.conf_int(), citation_count_model.pvalues], axis=1)
    model_result.reset_index(inplace=True)
    model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    model_result.to_csv("../results/covid19_linreg_citation_count_exported.csv", index=False)

    # Curvilinear effect of topic familiarity
    curvilinear_model = linreg_curvilinear_model(covid19_paper_scaled_df, [predictor_var] + control_var, "citation_count_log", "topic_familiarity", "covid19_linreg_curvilinear_tf_result_citation_count")
    model_result = pd.concat([curvilinear_model.params, curvilinear_model.bse, curvilinear_model.conf_int(), curvilinear_model.pvalues], axis=1)
    model_result.reset_index(inplace=True)
    model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    model_result.to_csv("../results/covid19_curvilinear_citation_count_exported.csv")




