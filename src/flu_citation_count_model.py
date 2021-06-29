import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot  as plt
from scipy.stats import pearsonr
from scipy import stats
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
    dropped = True
    while dropped:
        variables = X.columns
        dropped = False
        vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
        max_vif = max(vif)
        if max_vif > thresh:
            maxloc = vif.index(max_vif)
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

    features = [col.replace(col, "C({})".format(col)) if "publish_" in col else col for col in aftervif.columns]
    formula = "{} ~ {}".format(target_colname, "+".join(features))


    model = smf.ols(formula=formula, data=df).fit()
    with open("../results/updated_results/{}.csv".format(write_filename), "w") as fh:
        fh.write(model.summary().as_csv())
        fh.close()

    return model

def linreg_curvilinear_model(df, feature_colnames, target_colname, curvilinear_var, write_filename):
    X = df[feature_colnames]
    aftervif = calculate_vif(X)

    features = [col.replace(col, "C({})".format(col)) if "publish_" in col else col for col in aftervif.columns]
    formula = "{} ~ I({}**2) + {}".format(target_colname, curvilinear_var, "+".join(features))

    model = smf.ols(formula=formula, data=df).fit()

    with open("../results/updated_results/{}.csv".format(write_filename), "w") as fh:
        fh.write(model.summary().as_csv())

    fh.close()

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
    citation_count_log_transformed = np.log(flu_paper_df['citation_count']+1)
    citation_per_month_log_transformed = np.log(flu_paper_df['citation_per_month']+1)

    flu_paper_df = flu_paper_df.assign(team_size_log = team_size_log_transformed)
    flu_paper_df = flu_paper_df.assign(max_hindex_log = max_hindex_log_transformed)
    flu_paper_df = flu_paper_df.assign(citation_count_log = citation_count_log_transformed)
    flu_paper_df = flu_paper_df.assign(citation_per_month_log = citation_per_month_log_transformed)

    # get dummy variables for publish month
    publish_month_dummy = pd.get_dummies(flu_paper_df.publish_month_text)
    publish_month_dummy.columns = ["publish_{}".format(c) for c in publish_month_dummy.columns]
    flu_paper_df = flu_paper_df.assign(**publish_month_dummy)
    publish_month_columns = list(publish_month_dummy.columns)
    publish_month_columns.remove("publish_Sep")
    publish_month_columns.remove("publish_Aug")

    # Export citation count distribution
    citation_distr = flu_paper_df.citation_count
    citation_distr.to_csv("../results/flu_citation_distribution.csv", index=False)

    # define variables
    control_var = ["avg_tdsim", "new_tie_rate", "hindex_gini","cultural_similarity", "topic_familiarity_var", "max_hindex_log", "team_size_log", "prac_affil_rate"] + publish_month_columns + ["topic_distr{}".format(i) for i in range(1,20)]
    predictor_var = "topic_familiarity"
    target_var = "citation_per_month_log"

    # drop na rows
    flu_paper_df = flu_paper_df.dropna(subset=[predictor_var] + control_var + [target_var])
    print("Number of instances: {}".format(flu_paper_df.shape[0]))

    # Standardize
    standardize_columns = [predictor_var] + ["avg_tdsim", "new_tie_rate", "hindex_gini","cultural_similarity", "topic_familiarity_var", "max_hindex_log", "team_size_log", "prac_affil_rate", "days_passed", "novelty_10perc"] + ["topic_distr{}".format(i) for i in range(1,20)]
    X = np.array(flu_paper_df[standardize_columns])
    X_scaled = scale(X)

    flu_paper_scaled_df = flu_paper_df.copy()
    flu_paper_scaled_df[standardize_columns] = X_scaled

    # correlation matrix
    correlation_matrix = flu_paper_df[[predictor_var] + control_var + ["citation_count_log", "citation_counts_binary_5perc", "novelty_10perc"]].corr()
    correlation_matrix.to_csv("../results/updated_results/flu_modeling_variables_corr_matrix.csv")
    
    # print the summary stsatistics for each variable
    print("Mean: ")
    print(flu_paper_df[[predictor_var] + control_var + ["citation_per_month_log", "citation_per_month_binary_5perc", "novelty_10perc"]].mean())
    print("SD: ")
    print(flu_paper_df[[predictor_var] + control_var + ["citation_per_month_log", "citation_per_month_binary_5perc", "novelty_10perc"]].std())

    print("Modeling...")
    citation_count_model = linreg_model(flu_paper_scaled_df, [predictor_var] + control_var, "citation_per_month_log", "flu_linreg_result_citation_per_month")
    model_result = pd.concat([citation_count_model.params, citation_count_model.bse, citation_count_model.conf_int(), citation_count_model.pvalues], axis=1)
    model_result.reset_index(inplace=True)
    model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    model_result.to_csv("../results/updated_results/flu_linreg_citation_count_exported.csv", index=False)

    # Curvilinear effect of topic familiarity
    curvilinear_model = linreg_curvilinear_model(flu_paper_scaled_df, [predictor_var] + control_var, "citation_per_month_log", "topic_familiarity", "flu_linreg_curvilinear_tf_result_citation_count")
    model_result = pd.concat([curvilinear_model.params, curvilinear_model.bse, curvilinear_model.conf_int(), curvilinear_model.pvalues], axis=1)
    model_result.reset_index(inplace=True)
    model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    model_result.to_csv("../results/updated_results/flu_curvilinear_citation_count_exported.csv")

    # export data for curvilinear effect plotting
    flu_paper_scaled_df[["topic_familiarity", "citation_per_month_log", "citation_per_month_binary_5perc", "novelty_10perc"]].to_csv("../results/updated_results/flu_curvilinear_effect_data_exported.csv", index=False)

    # collaboration rate vs. time
    corr, pval = stats.pearsonr(flu_paper_scaled_df.new_tie_rate, flu_paper_scaled_df.days_passed)
    print("New collaboration rate vs. time correlation: {} ({})".format(corr, pval))
    #######################################################################################
    ### Interaction experiments
    #######################################################################################
    # add time * knowledge continuity interaction term to the model
    feature_colnames = [predictor_var] + control_var + ["days_passed"]
    X = flu_paper_scaled_df[feature_colnames]
    aftervif = calculate_vif(X)

    features = [col.replace(col, "C({})".format(col)) if "publish_" in col else col for col in aftervif.columns]
    formula = "{} ~ days_passed:topic_familiarity + {} + days_passed".format(target_var, "+".join(features))

    inter_model = smf.ols(formula=formula, data=flu_paper_scaled_df).fit()
    inter_model_result = pd.concat([inter_model.params, inter_model.bse, inter_model.conf_int(), inter_model.pvalues], axis=1)
    inter_model_result.reset_index(inplace=True)
    inter_model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    inter_model_result.to_csv("../results/updated_results/flu_time_kc_inter_citation_per_month_exported.csv")

    ## add novelty * knowledge continuity interaction term to the model
    flu_paper_scaled_df = flu_paper_scaled_df.dropna(subset=["novelty_10perc"]) # drop na rows

    feature_colnames = [predictor_var] + control_var + ["novelty_10perc"]
    X = flu_paper_scaled_df[feature_colnames]
    aftervif = calculate_vif(X)

    features = [col.replace(col, "C({})".format(col)) if "publish_" in col else col for col in aftervif.columns]
    formula = "{} ~ novelty_10perc:topic_familiarity + {} + novelty_10perc".format(target_var, "+".join(features))

    inter_model2 = smf.ols(formula=formula, data=flu_paper_scaled_df).fit()
    write_filename = "flu_linreg_novelty_kc_interaction_impact"
    with open("../results/updated_results/{}.csv".format(write_filename), "w") as fh:
        fh.write(inter_model2.summary().as_csv())

    # export dataset for interaction term plotting.
    flu_paper_scaled_df[["topic_familiarity", "days_passed", "novelty_10perc", "citation_per_month_log", "citation_per_month_binary_20perc", \
            "citation_per_month_binary_10perc", "citation_per_month_binary_5perc"]].to_csv("../results/updated_results/flu_scaled_interaction_exported.csv", index=False)

    ### Simple correlation analysis
    # novelty score vs. time
    corr, pval = stats.pearsonr(flu_paper_scaled_df.novelty_10perc, flu_paper_scaled_df.days_passed)
    print("Novelty score vs. time correlation: {} ({})".format(corr, pval))
