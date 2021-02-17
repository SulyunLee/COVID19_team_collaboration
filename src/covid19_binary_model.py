
import patsy
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

    target_colname = "citation_counts_binary_{}perc".format(int(perc*100))
    formula = "{} ~ {}".format(target_colname, "+".join(aftervif.columns))
    model = smf.logit(formula=formula, data=covid19_paper_scaled_df).fit()

    write_filename = "covid19_logit_model"
    with open("../results/{}_top{}%.csv".format(write_filename, int(perc*100)), "w") as fh:
        fh.write(model.summary().as_csv())

    return model

def fit_curvilinear_binary_model(perc, df, after_vif, curvilinear_var):
    target_colname = "citation_counts_binary_{}perc".format(int(perc*100))

    formula = "{} ~ I({}**2) + {}".format(target_colname, curvilinear_var, "+".join(aftervif.columns))
    # y, X = patsy.dmatrices(formula, df, return_type="matrix")
    model = smf.logit(formula=formula, data=df).fit()

    write_filename = "covid19_logit_curvilinear_{}_model".format(curvilinear_var)
    with open("../results/{}_top{}%.csv".format(write_filename, int(perc*100)), "w") as fh:
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


    # replace average topic distr similarity -1 as nan
    # value of -1 means zero or one authors have abstracts available
    avg_tdsim_replaced = covid19_paper_df['avg_tdsim'].replace(-1, np.nan)
    covid19_paper_df = covid19_paper_df.assign(avg_tdsim = avg_tdsim_replaced)

    team_size_log_transformed = np.log(covid19_paper_df['team_size']+1)
    max_hindex_log_transformed = np.log(covid19_paper_df['max_hindex']+1)
    citation_count_log_transformed = np.log(covid19_paper_df['citation_count']+1)

    covid19_paper_df = covid19_paper_df.assign(team_size_log = team_size_log_transformed)
    covid19_paper_df = covid19_paper_df.assign(max_hindex_log = max_hindex_log_transformed)
    covid19_paper_df = covid19_paper_df.assign(citation_count_log = citation_count_log_transformed)

    # Generate binary DVs
    covid19_paper_df = generate_binary_dv(0.2, covid19_paper_df)
    covid19_paper_df = generate_binary_dv(0.1, covid19_paper_df)
    covid19_paper_df = generate_binary_dv(0.05, covid19_paper_df)
    covid19_paper_df = generate_binary_dv(0.01, covid19_paper_df)

    # define variables
    control_var = ["avg_tdsim", "new_tie_rate", "hindex_gini","cultural_similarity", "topic_familiarity_var", "max_hindex_log", "time_since_pub", "team_size_log", "prac_affil_rate"] + ["topic_distr{}".format(i) for i in range(1,20)]
    predictor_var = "topic_familiarity"
    target_var = "citaion_counts_binary"

    # drop na rows
    covid19_paper_df = covid19_paper_df.dropna(subset=[predictor_var] + control_var)

    print("Number of instances: {}".format(covid19_paper_df.shape[0]))

    # Standardize
    X = np.array(covid19_paper_df[[predictor_var] + control_var])
    X_scaled = scale(X)

    covid19_paper_scaled_df = covid19_paper_df.copy()
    covid19_paper_scaled_df[[predictor_var] + control_var] = X_scaled

    # remove multi-collinearity
    X = covid19_paper_scaled_df[[predictor_var] + control_var]
    aftervif = calculate_vif(X)

    # Breakthrough - 20%
    model_20perc = fit_binary_model(0.2, covid19_paper_scaled_df, aftervif)
    model_result = pd.concat([model_20perc.params, model_20perc.bse, model_20perc.conf_int(), model_20perc.pvalues], axis=1)
    model_result.reset_index(inplace=True)
    model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    model_result.to_csv("../results/covid19_logit_breakthrough_20perc_exported.csv", index=False)

    # Breakthrough - 10%
    model_10perc = fit_binary_model(0.1, covid19_paper_scaled_df, aftervif)
    model_result = pd.concat([model_10perc.params, model_10perc.bse, model_10perc.conf_int(), model_10perc.pvalues], axis=1)
    model_result.reset_index(inplace=True)
    model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    model_result.to_csv("../results/covid19_logit_breakthrough_10perc_exported.csv", index=False)

    # Breakthrough - 5%
    model_5perc = fit_binary_model(0.05, covid19_paper_scaled_df, aftervif)
    model_result = pd.concat([model_5perc.params, model_5perc.bse, model_5perc.conf_int(), model_5perc.pvalues], axis=1)
    model_result.reset_index(inplace=True)
    model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    model_result.to_csv("../results/covid19_logit_breakthrough_5perc_exported.csv", index=False)

    # Breakthrough - 1%
    model_1perc = fit_binary_model(0.01, covid19_paper_scaled_df, aftervif)
    model_result = pd.concat([model_1perc.params, model_1perc.bse, model_1perc.conf_int(), model_1perc.pvalues], axis=1)
    model_result.reset_index(inplace=True)
    model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    model_result.to_csv("../results/covid19_logit_breakthrough_1perc_exported.csv", index=False)

    # Curvilinear model
    curvilinear_model_5perc = fit_curvilinear_binary_model(0.05, covid19_paper_scaled_df, aftervif, "topic_familiarity")
    model_result = pd.concat([curvilinear_model_5perc.params, curvilinear_model_5perc.bse, curvilinear_model_5perc.conf_int(), curvilinear_model_5perc.pvalues], axis=1)
    model_result.reset_index(inplace=True)
    model_result.columns = ["variable", "coef", "Std_err", "ci_low", "ci_high", "pval"]
    model_result.to_csv("../results/covid19_curvilinear_breakthrough_5perc_exported.csv", index=False)

