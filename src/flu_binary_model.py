

import patsy
import pandas as pd
import numpy as np
import random
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
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

def generate_binary_dv(perc, df):
    cut_point = int(df.shape[0]*perc)
    cut = df.citation_count_log.sort_values().iloc[-cut_point]
    citation_counts_binary = np.where(df.citation_count_log > cut, 1, 0)
    df['citation_counts_binary_{}perc'.format(int(perc*100))] = citation_counts_binary

    return df


def fit_binary_model(perc, flu_paper_scaled_df, aftervif):

    target_colname = "citation_per_month_binary_{}perc".format(int(perc*100))
    features = [col.replace(col, 'C({})'.format(col)) if "publish_" in col else col for col in aftervif.columns]
    formula = "{} ~ {}".format(target_colname, "+".join(features))
    y,x = dmatrices(formula, flu_paper_scaled_df, return_type="dataframe")

    # model = smf.logit(formula=formula, data=df).fit(method='lbfgs', maxiter=1000)
    model = sm.Logit(y, x).fit(maxiter=2000, method="lbfgs", retall=False)

    write_filename = "flu_logit_model"
    with open("../results/updated_results/{}_top{}%.csv".format(write_filename, int(perc*100)), "w") as fh:
        fh.write(model.summary().as_csv())
    fh.close()

    return model

def fit_curvilinear_binary_model(perc, df, after_vif, curvilinear_var):
    target_colname = "citation_per_month_binary_{}perc".format(int(perc*100))
    features = [col.replace(col, 'C({})'.format(col)) if "publish_" in col else col for col in aftervif.columns]

    formula = "{} ~ I({}**2) + {}".format(target_colname, curvilinear_var, "+".join(features))
    y, x = patsy.dmatrices(formula, df, return_type="dataframe")
    # model = smf.logit(formula=formula, data=df).fit(method='lbfgs', maxiter=1000)

    model = sm.Logit(y, x).fit(maxiter=2000, method="lbfgs", retall=False)

    write_filename = "flu_logit_curvilinear_{}_model".format(curvilinear_var)
    with open("../results/updated_results/{}_top{}%.csv".format(write_filename, int(perc*100)), "w") as fh:
        fh.write(model.summary().as_csv())
    fh.close()

    return model

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

    # get dummy variables for publish month
    publish_month_dummy = pd.get_dummies(flu_paper_df.publish_month_text)
    publish_month_dummy.columns = ["publish_{}".format(c) for c in publish_month_dummy.columns]
    flu_paper_df = flu_paper_df.assign(**publish_month_dummy)
    publish_month_columns = list(publish_month_dummy.columns)
    publish_month_columns.remove("publish_Sep")
    publish_month_columns.remove("publish_Aug")

    # Generate binary DVs
    # flu_paper_df = generate_binary_dv(0.2, flu_paper_df)
    # flu_paper_df = generate_binary_dv(0.1, flu_paper_df)
    # flu_paper_df = generate_binary_dv(0.05, flu_paper_df)
    # flu_paper_df = generate_binary_dv(0.01, flu_paper_df)

    # define variables
    control_var = ["avg_tdsim", "new_tie_rate", "hindex_gini","cultural_similarity", "topic_familiarity_var", "max_hindex_log", "team_size_log", "prac_affil_rate"] + publish_month_columns + ["topic_distr{}".format(i) for i in range(1,20)]
    predictor_var = "topic_familiarity"
    target_var = "citation_per_month_binary"

    # drop na rows
    flu_paper_df = flu_paper_df.dropna(subset=[predictor_var] + control_var + ["citation_per_month", "citation_per_month_binary_10perc", "citation_per_month_binary_5perc", "citation_per_month_binary_1perc"])
    print("Number of instances: {}".format(flu_paper_df.shape[0]))

    # Standardize
    standardize_columns = [predictor_var] + ["avg_tdsim", "new_tie_rate", "hindex_gini","cultural_similarity", "topic_familiarity_var", "max_hindex_log", "team_size_log", "prac_affil_rate", "days_passed", "novelty_10perc"] + ["topic_distr{}".format(i) for i in range(1,20)]
    X = np.array(flu_paper_df[standardize_columns])
    X_scaled = scale(X)

    flu_paper_scaled_df = flu_paper_df.copy()
    flu_paper_scaled_df[standardize_columns] = X_scaled

    # remove multi-collinearity
    X = flu_paper_scaled_df[[predictor_var] + control_var]
    aftervif = calculate_vif(X)

    # Breakthrough - 20%
    model_20perc = fit_binary_model(0.2, flu_paper_scaled_df, aftervif)
    model_result = pd.concat([model_20perc.params, model_20perc.bse, model_20perc.conf_int(), model_20perc.pvalues], axis=1)
    model_result.reset_index(inplace=True)
    model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    model_result.to_csv("../results/updated_results/flu_logit_breakthrough_20perc_exported.csv", index=False)

    # Breakthrough - 10%
    model_10perc = fit_binary_model(0.1, flu_paper_scaled_df, aftervif)
    model_result = pd.concat([model_10perc.params, model_10perc.bse, model_10perc.conf_int(), model_10perc.pvalues], axis=1)
    model_result.reset_index(inplace=True)
    model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    model_result.to_csv("../results/updated_results/flu_logit_breakthrough_10perc_exported.csv", index=False)

    model_5perc = fit_binary_model(0.05, flu_paper_scaled_df, aftervif)
    model_result = pd.concat([model_5perc.params, model_5perc.bse, model_5perc.conf_int(), model_5perc.pvalues], axis=1)
    model_result.reset_index(inplace=True)
    model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    model_result.to_csv("../results/updated_results/flu_logit_breakthrough_5perc_exported.csv", index=False)

    # Breakthrough - 1%
    # Use solver "cg" for 1% (due to class unbalance)
    model_1perc = fit_binary_model(0.01, flu_paper_scaled_df, aftervif)
    target_colname = "citation_per_month_binary_1perc"
    features = [col.replace(col, 'C({})'.format(col)) if "publish_" in col else col for col in aftervif.columns]
    formula = "{} ~ {}".format(target_colname, "+".join(features))
    y,x = dmatrices(formula, flu_paper_scaled_df, return_type="dataframe")

    model_1perc = sm.Logit(y, x).fit(maxiter=2000, method="cg", retall=False)

    write_filename = "flu_logit_model"
    with open("../results/updated_results/flu_logit_model_top1%.csv", "w") as fh:
        fh.write(model_1perc.summary().as_csv())
    fh.close()

    model_result = pd.concat([model_1perc.params, model_1perc.bse, model_1perc.conf_int(), model_1perc.pvalues], axis=1)
    model_result.reset_index(inplace=True)
    model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    model_result.to_csv("../results/updated_results/flu_logit_breakthrough_1perc_exported.csv", index=False)

    # Curvilinear model
    curvilinear_model_5perc = fit_curvilinear_binary_model(0.05, flu_paper_scaled_df, aftervif, "topic_familiarity")
    model_result = pd.concat([curvilinear_model_5perc.params, curvilinear_model_5perc.bse, curvilinear_model_5perc.conf_int(), curvilinear_model_5perc.pvalues], axis=1)
    model_result.reset_index(inplace=True)
    model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    model_result.to_csv("../results/updated_results/flu_curvilinear_breakthrough_5perc_exported.csv", index=False)

    #######################################################################################
    ### Interaction experiments
    #######################################################################################
    # add time * knowledge continuity interaction term to the model
    feature_colnames = [predictor_var] + control_var + ["days_passed"]
    X = flu_paper_scaled_df[feature_colnames]
    aftervif = calculate_vif(X)

    features = [col.replace(col, "C({})".format(col)) if "publish_" in col else col for col in aftervif.columns]
    target_var = "citation_per_month_binary_5perc"
    formula = "{} ~ days_passed:topic_familiarity + {} + days_passed".format(target_var, "+".join(features))

    y,x = dmatrices(formula, flu_paper_scaled_df, return_type="dataframe")

    inter_model = sm.Logit(y, x).fit(maxiter=2000, method="lbfgs", retall=False)
    inter_model_result = pd.concat([inter_model.params, inter_model.bse, inter_model.conf_int(), inter_model.pvalues], axis=1)
    inter_model_result.reset_index(inplace=True)
    inter_model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    inter_model_result.to_csv("../results/updated_results/flu_time_kc_inter_citation_per_month_binary_5perc_exported.csv")

    ## add novelty * knowledge continuity interaction term to the model
    flu_paper_scaled_df = flu_paper_scaled_df.dropna(subset=["novelty_10perc"])
    feature_colnames = [predictor_var] + control_var + ["novelty_10perc"]
    X = flu_paper_scaled_df[feature_colnames]
    aftervif = calculate_vif(X)

    features = [col.replace(col, "C({})".format(col)) if "publish_" in col else col for col in aftervif.columns]
    target_var = "citation_per_month_binary_5perc"
    formula = "{} ~ novelty_10perc:topic_familiarity + {} + novelty_10perc".format(target_var, "+".join(features))

    y,x = dmatrices(formula, flu_paper_scaled_df, return_type="dataframe")

    inter_model2 = sm.Logit(y, x).fit(maxiter=2000, method="lbfgs", retall=False)

    write_filename = "flu_logit_novelty_kc_interaction_breakthrough"
    with open("../results/updated_results/{}_top5%.csv".format(write_filename), "w") as fh:
        fh.write(inter_model2.summary().as_csv())





