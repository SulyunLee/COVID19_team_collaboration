import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot  as plt
from scipy.stats import pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import norm
from scipy import stats
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

    return model

def linreg_curvilinear_model(df, feature_colnames, target_colname, curvilinear_var, write_filename):
    X = df[feature_colnames]
    aftervif = calculate_vif(X)

    features = [col.replace(col, "C({})".format(col)) if "publish_" in col else col for col in aftervif.columns]
    formula = "{} ~ I({}**2) + {}".format(target_colname, curvilinear_var, "+".join(aftervif.columns))

    model = smf.ols(formula=formula, data=df).fit()

    with open("../results/updated_results/{}.csv".format(write_filename), "w") as fh:
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
    citation_per_month_log_transformed = np.log(covid19_paper_df['citation_per_month']+1)

    covid19_paper_df = covid19_paper_df.assign(team_size_log = team_size_log_transformed)
    covid19_paper_df = covid19_paper_df.assign(max_hindex_log = max_hindex_log_transformed)
    covid19_paper_df = covid19_paper_df.assign(citation_count_log = citation_count_log_transformed)
    covid19_paper_df = covid19_paper_df.assign(mention_log = mention_log_transformed)
    covid19_paper_df = covid19_paper_df.assign(citation_per_month_log = citation_per_month_log_transformed)

    author_id_eval = covid19_paper_df['author_id_modified'].apply(lambda x: eval(x))
    covid19_paper_df = covid19_paper_df.assign(author_id_modified=author_id_eval)

    # get dummy variables for publish month
    publish_month_dummy = pd.get_dummies(covid19_paper_df.publish_month_text)
    publish_month_dummy.columns = ["publish_{}".format(c) for c in publish_month_dummy.columns]
    covid19_paper_df = covid19_paper_df.assign(**publish_month_dummy)
    publish_month_columns = list(publish_month_dummy.columns)
    publish_month_columns.remove("publish_Aug")
    publish_month_columns.remove("publish_Sep")

    # Export citation count distribution
    citation_distr = covid19_paper_df.citation_count
    citation_distr.to_csv("../results/covid19_citation_distribution.csv", index=False)

    # define variables
    control_var = ["avg_tdsim", "new_tie_rate", "hindex_gini","cultural_similarity", "topic_familiarity_var", "max_hindex_log", "team_size_log", "prac_affil_rate"] + publish_month_columns + ["topic_distr{}".format(i) for i in range(1,20)]
    predictor_var = "topic_familiarity"
    target_var = "citation_per_month_log"

    # drop na rows
    covid19_paper_df = covid19_paper_df.dropna(subset=[predictor_var] + control_var + [target_var])
    print("Number of instances: {}".format(covid19_paper_df.shape[0]))

    # Standardize
    standardize_columns = [predictor_var] + ["avg_tdsim", "new_tie_rate", "hindex_gini","cultural_similarity", "topic_familiarity_var", "max_hindex_log", "team_size_log", "prac_affil_rate", "days_passed", "novelty_10perc"] + ["topic_distr{}".format(i) for i in range(1,20)]
    X = np.array(covid19_paper_df[standardize_columns])
    X_scaled = scale(X)

    covid19_paper_scaled_df = covid19_paper_df.copy()
    covid19_paper_scaled_df[standardize_columns] = X_scaled

    # correlation matrix
    correlation_matrix = covid19_paper_df[[predictor_var] + control_var + ["citation_per_month_log", "citation_per_month_binary_5perc", "novelty_10perc"]].corr()
    correlation_matrix.to_csv("../results/updated_results/covid19_modeling_variables_corr_matrix.csv")

    # print the summary stsatistics for each variable
    print("Mean: ")
    print(covid19_paper_df[[predictor_var] + control_var + ["citation_per_month_log", "citation_per_month_binary_5perc", "novelty_10perc"]].mean())
    print("SD: ")
    print(covid19_paper_df[[predictor_var] + control_var + ["citation_per_month_log", "citation_per_month_binary_5perc", "novelty_10perc"]].std())

    print("Modeling...")
    citation_count_model = linreg_model(covid19_paper_scaled_df, [predictor_var] + control_var, "citation_per_month_log", "covid19_linreg_result_citation_per_month")
    model_result = pd.concat([citation_count_model.params, citation_count_model.bse, citation_count_model.conf_int(), citation_count_model.pvalues], axis=1)
    model_result.reset_index(inplace=True)
    model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    model_result.to_csv("../results/updated_results/covid19_linreg_citation_count_exported.csv", index=False)

    # Curvilinear effect of topic familiarity
    singlevar_curvilinear = smf.ols(formula="citation_per_month_log ~ I(topic_familiarity**2)", data=covid19_paper_scaled_df).fit()

    curvilinear_model = linreg_curvilinear_model(covid19_paper_scaled_df, [predictor_var] + control_var, "citation_per_month_log", "topic_familiarity", "covid19_linreg_curvilinear_tf_result_citation_per_month")
    model_result_curvilinear = pd.concat([curvilinear_model.params, curvilinear_model.bse, curvilinear_model.conf_int(), curvilinear_model.pvalues], axis=1)
    model_result_curvilinear.reset_index(inplace=True)
    model_result_curvilinear.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    model_result_curvilinear.to_csv("../results/updated_results/covid19_curvilinear_citation_count_exported.csv")

    # export data for curvilinear effect plotting
    covid19_paper_scaled_df[["topic_familiarity", "citation_per_month_log", "citation_per_month_binary_5perc", "novelty_10perc"]].to_csv("../results/updated_results/covid19_curvilinear_effect_data_exported.csv", index=False)

    # collaboration rate vs. time
    corr, pval = stats.pearsonr(covid19_paper_scaled_df.new_tie_rate, covid19_paper_scaled_df.days_passed)
    print("New collaboration rate vs. time correlation: {} ({})".format(corr, pval))

    ob_range = covid19_paper_scaled_df[(covid19_paper_scaled_df.topic_familiarity >= -1) & (covid19_paper_scaled_df.topic_familiarity <= 1)]
    breakthrough_over_apex = ob_range[ob_range.topic_familiarity > 0.5].shape[0]
    print("Breakthrough over apex: {}".format(breakthrough_over_apex/ob_range.shape[0]))
    #######################################################################################
    ### Interaction experiments
    #######################################################################################
    ## add time * knowledge continuity interaction term to the model
    print("## Time * Knowledge continuity interaction ##")
    feature_colnames = [predictor_var] + control_var + ["days_passed"]
    X = covid19_paper_scaled_df[feature_colnames]
    aftervif = calculate_vif(X)

    features = [col.replace(col, "C({})".format(col)) if "publish_" in col else col for col in aftervif.columns]
    formula = "{} ~ days_passed:topic_familiarity + {} + days_passed".format(target_var, "+".join(features))

    inter_model = smf.ols(formula=formula, data=covid19_paper_scaled_df).fit()
    inter_model_result = pd.concat([inter_model.params, inter_model.bse, inter_model.conf_int(), inter_model.pvalues], axis=1)
    inter_model_result.reset_index(inplace=True)
    inter_model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    inter_model_result.to_csv("../results/updated_results/covid19_time_kc_inter_citation_per_month_exported.csv")

    # region of significance test
    # Used program here: http://quantpsy.org/interact/mlr2.htm#:~:text=The%20region%20of%20significance%20defines,upper%20bounds%20to%20the%20region.
    print("Regression coefficients: ")
    print(inter_model.params.loc[["Intercept","topic_familiarity","days_passed","days_passed:topic_familiarity"],])

    print("Coefficient variances: ")
    print(inter_model.bse.loc[["Intercept","topic_familiarity","days_passed","days_passed:topic_familiarity"],] ** 2)

    print("Coefficient covariances: ")
    print(inter_model.cov_params().loc["days_passed","Intercept"], inter_model.cov_params().loc["days_passed:topic_familiarity", "topic_familiarity"])

    ## add novelty * knowledge continuity interaction term to the model
    print("## Novelty * Knowledge continuity interaction ##")
    covid19_paper_scaled_df = covid19_paper_scaled_df.dropna(subset=["novelty_10perc"]) # drop na rows

    ob_range = covid19_paper_scaled_df[(covid19_paper_scaled_df.topic_familiarity >= -1) & (covid19_paper_scaled_df.topic_familiarity <= 1)]
    novelty_over_apex = ob_range[ob_range.topic_familiarity > 0.8].shape[0]
    print("Novelty over apex: {}".format(novelty_over_apex/ob_range.shape[0]))

    feature_colnames = [predictor_var] + control_var + ["novelty_10perc"]
    X = covid19_paper_scaled_df[feature_colnames]
    aftervif = calculate_vif(X)

    features = [col.replace(col, "C({})".format(col)) if "publish_" in col else col for col in aftervif.columns]
    formula = "{} ~ novelty_10perc:topic_familiarity + {} + novelty_10perc".format(target_var, "+".join(features))

    inter_model2 = smf.ols(formula=formula, data=covid19_paper_scaled_df).fit()
    write_filename = "covid19_linreg_novelty_kc_interaction_impact"
    with open("../results/updated_results/{}.csv".format(write_filename), "w") as fh:
        fh.write(inter_model2.summary().as_csv())

    # region of significance test
    print("Regression coefficients: ")
    print(inter_model2.params.loc[["Intercept","topic_familiarity","novelty_10perc","novelty_10perc:topic_familiarity"],])

    print("Coefficient variances: ")
    print(inter_model2.bse.loc[["Intercept","topic_familiarity","novelty_10perc","novelty_10perc:topic_familiarity"],] ** 2)

    print("Coefficient covariances: ")
    print(inter_model2.cov_params().loc["novelty_10perc","Intercept"], inter_model2.cov_params().loc["novelty_10perc:topic_familiarity", "topic_familiarity"])

    # export dataset for interaction term plotting.
    covid19_paper_scaled_df[["topic_familiarity", "days_passed", "novelty_10perc", "citation_per_month_log", "citation_per_month_binary_20perc", \
            "citation_per_month_binary_10perc", "citation_per_month_binary_5perc"]].to_csv("../results/updated_results/covid19_scaled_interaction_exported.csv", index=False)


    ### Simple correlation analysis
    # novelty score vs. time
    corr, pval = stats.pearsonr(covid19_paper_scaled_df.novelty_10perc, covid19_paper_scaled_df.days_passed)
    print("Novelty score vs. time correlation: {} ({})".format(corr, pval))




