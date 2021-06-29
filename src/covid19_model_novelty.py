
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
    formula = "{} ~ {}".format(target_colname, "+".join(aftervif.columns))


    model = smf.ols(formula=formula, data=df).fit()
    with open("../results/updated_results/{}.csv".format(write_filename), "w") as fh:
        fh.write(model.summary().as_csv())
    fh.close()

    return model

def linreg_curvilinear_model(df, feature_colnames, target_colname, curvilinear_var, write_filename):
    X = df[feature_colnames]
    aftervif = calculate_vif(X)

    features = [col.replace(col, "C({})".format(col)) if "publish_" in col else col for col in aftervif.columns]
    formula = "{} ~ I({}**2) + {}".format(target_colname, curvilinear_var, "+".join(aftervif.columns))

    model = smf.ols(formula=formula, data=df).fit()

    with open("../results/updated_results/{}.csv".format(write_filename), "w") as fh:
        fh.write(model.summary().as_csv())
    fh.close()

    return model

if __name__ == "__main__":
    print("Loading modeling data...")
    covid19_papers_modeling_filename = "../dataset/COVID19_papers_modeling_with_features.csv"
    covid19_paper_df = pd.read_csv(covid19_papers_modeling_filename)

    print("Cleaning data...")
    citation_count_replaced = covid19_paper_df['citation_count'].replace(-1,np.nan)
    covid19_paper_df = covid19_paper_df.assign(citation_count = citation_count_replaced)

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

    # get dummy variables for publish month
    publish_month_dummy = pd.get_dummies(covid19_paper_df.publish_month_text)
    publish_month_dummy.columns = ["publish_{}".format(c) for c in publish_month_dummy.columns]
    covid19_paper_df = covid19_paper_df.assign(**publish_month_dummy)
    publish_month_columns = list(publish_month_dummy.columns)
    publish_month_columns.remove("publish_Aug")
    publish_month_columns.remove("publish_Sep")

    # define variables
    control_var = ["avg_tdsim", "new_tie_rate", "hindex_gini","cultural_similarity", "topic_familiarity_var", "max_hindex_log", "team_size_log", "prac_affil_rate"] + publish_month_columns + ["topic_distr{}".format(i) for i in range(1,20)]
    predictor_var = "topic_familiarity"

    # drop na rows
    covid19_paper_df = covid19_paper_df.dropna(subset=[predictor_var] + control_var + ["novelty_10perc", "novelty_5perc", "novelty_1perc"])
    print("Number of instances: {}".format(covid19_paper_df.shape[0]))

    # plot novelty variable
    # sns.distplot(covid19_paper_df['novelty_10perc'])
    # plt.title("Distribution of novelty (10%)")
    # plt.savefig("../plots/covid19_novelty_10perc.png")
    # plt.close()

    # sns.distplot(covid19_paper_df['novelty_5perc'])
    # plt.title("Distribution of novelty (5%)")
    # plt.savefig("../plots/covid19_novelty_5perc.png")
    # plt.close()

    # sns.distplot(covid19_paper_df['novelty_1perc'])
    # plt.title("Distribution of novelty (1%)")
    # plt.savefig("../plots/covid19_novelty_1perc.png")
    # plt.close()

    # Standardize
    standardize_columns = [predictor_var] + ["avg_tdsim", "new_tie_rate", "hindex_gini","cultural_similarity", "topic_familiarity_var", "max_hindex_log", "team_size_log", "prac_affil_rate", "days_passed", "novelty_10perc"] + ["topic_distr{}".format(i) for i in range(1,20)]
    X = np.array(covid19_paper_df[standardize_columns])
    X_scaled = scale(X)

    covid19_paper_scaled_df = covid19_paper_df.copy()
    covid19_paper_scaled_df[standardize_columns] = X_scaled

    print("Modeling..")
    # Novelty - 10%
    novelty_10perc_model = linreg_model(covid19_paper_scaled_df, [predictor_var] + control_var, "novelty_10perc", "covid19_linreg_result_novelty_10perc")
    model_result = pd.concat([novelty_10perc_model.params, novelty_10perc_model.bse, novelty_10perc_model.conf_int(), novelty_10perc_model.pvalues], axis=1)
    model_result.reset_index(inplace=True)
    model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    model_result.to_csv("../results/updated_results/covid19_linreg_novelty_10perc_exported.csv", index=False)

    # Novelty - 5%
    novelty_5perc_model = linreg_model(covid19_paper_scaled_df, [predictor_var] + control_var, "novelty_5perc", "covid19_linreg_result_novelty_5perc")
    model_result = pd.concat([novelty_5perc_model.params, novelty_5perc_model.bse, novelty_5perc_model.conf_int(), novelty_5perc_model.pvalues], axis=1)
    model_result.reset_index(inplace=True)
    model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    model_result.to_csv("../results/updated_results/covid19_linreg_novelty_5perc_exported.csv", index=False)

    # Novelty - 1%
    novelty_1perc_model = linreg_model(covid19_paper_scaled_df, [predictor_var] + control_var, "novelty_1perc", "covid19_linreg_result_novelty_1perc")
    model_result = pd.concat([novelty_1perc_model.params, novelty_1perc_model.bse, novelty_1perc_model.conf_int(), novelty_1perc_model.pvalues], axis=1)
    model_result.reset_index(inplace=True)
    model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    model_result.to_csv("../results/updated_results/covid19_linreg_novelty_1perc_exported.csv", index=False)


    # Curvilinear model
    curvilinear_model = linreg_curvilinear_model(covid19_paper_scaled_df, [predictor_var] + control_var, "novelty_10perc", "topic_familiarity", "covid19_linreg_curvilinear_tf_result_novelty_10perc")
    model_result = pd.concat([curvilinear_model.params, curvilinear_model.bse, curvilinear_model.conf_int(), curvilinear_model.pvalues], axis=1)
    model_result.reset_index(inplace=True)
    model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    model_result.to_csv("../results/updated_results/covid19_curvilinear_novelty_10perc_exported.csv")


    ######################################################################################
    ### Interaction experiments
    #######################################################################################
    ## add time * knowledge continuity interaction term to the model
    print("## Time * Knowledge continuity interaction ##")
    target_var = "novelty_10perc"
    feature_colnames = [predictor_var] + control_var + ["days_passed"]
    X = covid19_paper_scaled_df[feature_colnames]
    aftervif = calculate_vif(X)

    features = [col.replace(col, "C({})".format(col)) if "publish_" in col else col for col in aftervif.columns]
    formula = "{} ~ days_passed:topic_familiarity + {} + days_passed".format(target_var, "+".join(features))

    inter_model = smf.ols(formula=formula, data=covid19_paper_scaled_df).fit()
    inter_model_result = pd.concat([inter_model.params, inter_model.bse, inter_model.conf_int(), inter_model.pvalues], axis=1)
    inter_model_result.reset_index(inplace=True)
    inter_model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    inter_model_result.to_csv("../results/updated_results/covid19_time_kc_inter_novelty10perc__exported.csv")
