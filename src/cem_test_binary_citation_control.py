

import pandas as pd
import numpy as np
import random
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import scale
from patsy import dmatrices

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


def cem_test(perc, var, combined_papers_scaled, feature_var):
    # remove multi-collinearity
    # aftervif = calculate_vif(combined_papers_scaled[feature_var])
    aftervif = combined_papers_scaled[feature_var]

    features = [col.replace(col, 'C({})'.format(col)) if "publish_" in col else col for col in aftervif.columns]

    formula = "citation_per_month_binary_{}perc ~ C(covid19) + C(covid19):{} + {}".format(int(perc*100), var, "+".join(features))
    y, x = dmatrices(formula, combined_papers_scaled, return_type="dataframe")

    model = sm.Logit(y, x).fit(maxiter=2000, method="lbfgs", retall=False)

    write_filename = "citation_binary_{}perc_{}_cem_model".format(int(perc*100), var)
    with open("../results/updated_results/{}.csv".format(write_filename), "w") as fh:
        fh.write(model.summary().as_csv())

    return model

if __name__ == "__main__":
    print("Loading modeling data...")
    # COVID19
    covid19_papers_modeling_filename = "../dataset/COVID19_papers_modeling_with_features.csv"
    covid19_paper_df = pd.read_csv(covid19_papers_modeling_filename)
    
    # flu
    flu_papers_modeling_filename = "../dataset/FLU_papers_modeling_with_features.csv"
    flu_paper_df = pd.read_csv(flu_papers_modeling_filename)

    print("Cleaning data...")
    # COVID19
    citation_count_replaced = covid19_paper_df['citation_count'].replace(-1,np.nan)
    covid19_paper_df = covid19_paper_df.assign(citation_count = citation_count_replaced)

    # replace average topic distr similarity -1 as nan
    # value of -1 means zero or one authors have abstracts available
    avg_tdsim_replaced = covid19_paper_df['avg_tdsim'].replace(-1, np.nan)
    covid19_paper_df = covid19_paper_df.assign(avg_tdsim = avg_tdsim_replaced)

    # flu paper
    cultural_sim_replaced = flu_paper_df['cultural_similarity'].replace(-1,np.nan)
    flu_paper_df = flu_paper_df.assign(cultural_similarity = cultural_sim_replaced)

    print("Log transforming...")
    # COVID19
    team_size_log_transformed = np.log(covid19_paper_df['team_size']+1)
    max_hindex_log_transformed = np.log(covid19_paper_df['max_hindex']+1)
    citation_count_log_transformed = np.log(covid19_paper_df['citation_count']+1)
    citation_per_month_log_transformed = np.log(covid19_paper_df['citation_per_month']+1)

    covid19_paper_df = covid19_paper_df.assign(team_size_log = team_size_log_transformed)
    covid19_paper_df = covid19_paper_df.assign(max_hindex_log = max_hindex_log_transformed)
    covid19_paper_df = covid19_paper_df.assign(citation_count_log = citation_count_log_transformed)
    covid19_paper_df = covid19_paper_df.assign(citation_per_month_log = citation_per_month_log_transformed)

    # flu
    team_size_log_transformed = np.log(flu_paper_df['team_size']+1)
    max_hindex_log_transformed = np.log(flu_paper_df['max_hindex']+1)
    citation_count_log_transformed = np.log(flu_paper_df['citation_count']+1)
    citation_per_month_log_transformed = np.log(flu_paper_df['citation_per_month']+1)

    flu_paper_df = flu_paper_df.assign(team_size_log = team_size_log_transformed)
    flu_paper_df = flu_paper_df.assign(max_hindex_log = max_hindex_log_transformed)
    flu_paper_df = flu_paper_df.assign(citation_count_log = citation_count_log_transformed)
    flu_paper_df = flu_paper_df.assign(citation_per_month_log = citation_per_month_log_transformed)

    # get dummy variables for publish month
    publish_month_dummy = pd.get_dummies(covid19_paper_df.publish_month_text)
    publish_month_dummy.columns = ["publish_{}".format(c) for c in publish_month_dummy.columns]
    covid19_paper_df = covid19_paper_df.assign(**publish_month_dummy)
    publish_month_columns = list(publish_month_dummy.columns)
    publish_month_columns.remove("publish_Aug")
    publish_month_columns.remove("publish_Sep")

    publish_month_dummy = pd.get_dummies(flu_paper_df.publish_month_text)
    publish_month_dummy.columns = ["publish_{}".format(c) for c in publish_month_dummy.columns]
    flu_paper_df = flu_paper_df.assign(**publish_month_dummy)
    publish_month_columns = list(publish_month_dummy.columns)
    publish_month_columns.remove("publish_Sep")
    publish_month_columns.remove("publish_Aug")

    # define variables
    control_var = ["avg_tdsim", "new_tie_rate", "hindex_gini","cultural_similarity", "topic_familiarity_var", "max_hindex_log", "team_size_log", "prac_affil_rate"] + publish_month_columns + ["topic_distr{}".format(i) for i in range(1,20)]
    predictor_var = "topic_familiarity"

    # drop NA rows
    # COVID19
    covid19_paper_df = covid19_paper_df.dropna(subset=[predictor_var] + control_var + ["citation_per_month_log", "citation_per_month_binary_10perc", "citation_per_month_binary_5perc", "citation_per_month_binary_1perc"])
    print("Number of COVID papers: {}".format(covid19_paper_df.shape[0]))

    # flu
    flu_paper_df = flu_paper_df.dropna(subset=[predictor_var] + control_var + ["citation_per_month_log", "citation_per_month_binary_10perc", "citation_per_month_binary_5perc", "citation_per_month_binary_1perc"])
    print("Number of flu papers: {}".format(flu_paper_df.shape[0]))

    print("Creating matching variable...")
    binning_control_var = ["avg_tdsim", "hindex_gini", "cultural_similarity", "topic_familiarity_var", "max_hindex_log", "team_size_log", "prac_affil_rate"]
    binning_control_var_round = covid19_paper_df[binning_control_var].round(2)
    covid19_paper_df = covid19_paper_df.assign(**binning_control_var_round)
    binning_control_var_round = flu_paper_df[binning_control_var].round(2)
    flu_paper_df = flu_paper_df.assign(**binning_control_var_round)

    ## Discretizing control variables - only use the common distributions 
    # Expertise similarity
    covid19_avg_tdsim_converted = covid19_paper_df.avg_tdsim.apply(lambda x: 0 if x >= 0 and x <= 0.5 else (1 if x > 0.5 and x <= 1 else np.nan))
    covid19_paper_df = covid19_paper_df.assign(avg_tdsim_binary=covid19_avg_tdsim_converted)
    flu_avg_tdsim_converted = flu_paper_df.avg_tdsim.apply(lambda x: 0 if x >= 0 and x <= 0.5 else (1 if x > 0.5 and x <= 1 else np.nan))
    flu_paper_df = flu_paper_df.assign(avg_tdsim_binary=flu_avg_tdsim_converted)

    # New collaboration rate
    # covid19_new_tie_rate_converted = covid19_paper_df.new_tie_rate.apply(lambda x: 0 if x == 0 else (1 if x >= 0.5 and x <= 1 else np.nan))
    # covid19_paper_df = covid19_paper_df.assign(new_tie_rate_binary=covid19_new_tie_rate_converted)
    # flu_new_tie_rate_converted = flu_paper_df.new_tie_rate.apply(lambda x: 0 if x == 0 else (1 if x >= 0.5 and x <= 1 else np.nan))
    # flu_paper_df = flu_paper_df.assign(new_tie_rate_binary=flu_new_tie_rate_converted)

    # Disparity of h-index
    covid19_hindex_gini_converted = covid19_paper_df.hindex_gini.apply(lambda x: 0 if x >= 0 and x <= 0.5 else (1 if x > 0.5 and x <= 1 else np.nan))
    covid19_paper_df = covid19_paper_df.assign(hindex_gini_binary=covid19_hindex_gini_converted)
    flu_hindex_gini_converted = flu_paper_df.hindex_gini.apply(lambda x: 0 if x >= 0 and x <= 0.5 else (1 if x > 0.5 and x <= 1 else np.nan))
    flu_paper_df = flu_paper_df.assign(hindex_gini_binary=flu_hindex_gini_converted)

    # Similarity of cultural background
    covid19_cultural_similarity_converted = covid19_paper_df.cultural_similarity.apply(lambda x: 0 if x > 0.5 and x < 1 else (1 if x == 1 else np.nan))
    covid19_paper_df = covid19_paper_df.assign(cultural_similarity_binary = covid19_cultural_similarity_converted)
    flu_cultural_similarity_converted = flu_paper_df.cultural_similarity.apply(lambda x: 0 if x > 0.5 and x < 1 else (1 if x == 1 else np.nan))
    flu_paper_df = flu_paper_df.assign(cultural_similarity_binary = flu_cultural_similarity_converted)

    # Variance of knowledge continuity
    covid19_topic_familiarity_var_converted = covid19_paper_df.topic_familiarity_var.apply(lambda x: 0 if x >= 0 and x <= 0.1 else (1 if x > 0.1 and x <= 0.5 else np.nan))
    covid19_paper_df = covid19_paper_df.assign(topic_familiarity_var_binary = covid19_topic_familiarity_var_converted)
    flu_topic_familiarity_var_converted = flu_paper_df.topic_familiarity_var.apply(lambda x: 0 if x >= 0 and x <= 0.1 else (1 if x > 0.1 and x <= 0.5 else np.nan))
    flu_paper_df = flu_paper_df.assign(topic_familiarity_var_binary = flu_topic_familiarity_var_converted)

    # Maximum h-index (log)
    covid19_max_hindex_log_converted = covid19_paper_df.max_hindex_log.apply(lambda x: 0 if x >= 0 and x <= 2 else (1 if x > 2 and x <= 3 else (2 if x > 3 and x <= 4 else (3 if x > 4 and x <= 6 else np.nan))))
    covid19_paper_df = covid19_paper_df.assign(max_hindex_log_binary = covid19_max_hindex_log_converted)
    flu_max_hindex_log_converted = flu_paper_df.max_hindex_log.apply(lambda x: 0 if x >= 0 and x <= 2 else (1 if x > 2 and x <= 3 else (2 if x > 3 and x <= 4 else (3 if x > 4 and x <= 6 else np.nan))))
    flu_paper_df = flu_paper_df.assign(max_hindex_log_binary = flu_max_hindex_log_converted)

    # Team size (log)
    covid19_team_size_log_converted = covid19_paper_df.team_size_log.apply(lambda x: 0 if x >= 1 and x <= 2 else (1 if x > 2 and x <= 3 else (2 if x > 3 and x <= 5 else np.nan)))
    covid19_paper_df = covid19_paper_df.assign(team_size_log_binary = covid19_team_size_log_converted)
    flu_team_size_log_converted = flu_paper_df.team_size_log.apply(lambda x: 0 if x >= 1 and x <= 2 else (1 if x > 2 and x <= 3 else (2 if x > 3 and x <= 5 else np.nan)))
    flu_paper_df = flu_paper_df.assign(team_size_log_binary = flu_team_size_log_converted)

    # Rate of practical affiliation
    covid19_prac_affil_rate_converted = covid19_paper_df.prac_affil_rate.apply(lambda x: 0 if x == 0 else (1 if x > 0 and x <= 0.5 else (2 if x > 0.5 and x < 1 else (3 if x == 1 else np.nan))))
    covid19_paper_df = covid19_paper_df.assign(prac_affil_rate_binary = covid19_prac_affil_rate_converted)
    flu_prac_affil_rate_converted = flu_paper_df.prac_affil_rate.apply(lambda x: 0 if x == 0 else (1 if x > 0 and x <= 0.5 else (2 if x > 0.5 and x < 1 else (3 if x == 1 else np.nan))))
    flu_paper_df = flu_paper_df.assign(prac_affil_rate_binary = flu_prac_affil_rate_converted)

    # exclude samples with no discretization included
    covid19_paper_binned_df = covid19_paper_df.dropna(subset=["{}_binary".format(x) for x in binning_control_var])
    covid19_paper_binned_df.reset_index(drop=True, inplace=True)
    flu_paper_binned_df = flu_paper_df.dropna(subset=["{}_binary".format(x) for x in binning_control_var])
    flu_paper_binned_df.reset_index(drop=True, inplace=True)
    print("The number of samples after dropping non-binned instances: {} COVID-19 and {} flu samples".format(covid19_paper_binned_df.shape[0], flu_paper_binned_df.shape[0]))

    random.seed(100)

    # convert binned variables to integer type
    covid19_binned_var_integer = covid19_paper_binned_df[["{}_binary".format(x) for x in binning_control_var] + publish_month_columns].astype(int)
    covid19_paper_binned_df = covid19_paper_binned_df.assign(**covid19_binned_var_integer)
    flu_binned_var_integer = flu_paper_binned_df[["{}_binary".format(x) for x in binning_control_var] + publish_month_columns].astype(int)
    flu_paper_binned_df = flu_paper_binned_df.assign(**flu_binned_var_integer)

    # extract samples with matching variables
    matching_covid19_df = covid19_paper_binned_df[["{}_binary".format(x) for x in binning_control_var] + publish_month_columns]
    matching_flu_df = flu_paper_binned_df[["{}_binary".format(x) for x in binning_control_var] + publish_month_columns]

    matched_flu_scopusid = []
    matched_covid19_uid = []
    for idx, flu_paper in matching_flu_df.iterrows():
        binned_values = flu_paper.values
        bin_values_matched = (matching_covid19_df.values == binned_values)
        matched_covid19_paper_idx = np.where(bin_values_matched.sum(axis=1) == matching_covid19_df.shape[1])[0]
        if matched_covid19_paper_idx.shape[0] != 0:
            matched_covid19_paper = covid19_paper_binned_df.iloc[matched_covid19_paper_idx,:].cord_uid.values
            new_matched_covid19_paper = [p for p in matched_covid19_paper if p not in matched_covid19_uid]
            if len(new_matched_covid19_paper) != 0:
                sampled_covid19_match = random.choice(new_matched_covid19_paper)

                matched_flu_paper = flu_paper_binned_df.iloc[idx].scopus_id
                matched_flu_scopusid.append(matched_flu_paper)
                matched_covid19_uid.append(sampled_covid19_match)

    print("Generating covid indicator variable...")
    covid19_matched = covid19_paper_binned_df[covid19_paper_binned_df.cord_uid.isin(matched_covid19_uid)]
    covid19_matched.reset_index(drop=True, inplace=True)
    covid19_matched = covid19_matched.assign(covid19=int(1))

    flu_matched = flu_paper_binned_df[flu_paper_binned_df.scopus_id.isin(matched_flu_scopusid)]
    flu_matched.reset_index(drop=True, inplace=True)
    flu_matched = flu_matched.assign(covid19=int(0))

    covid19_matched = covid19_matched[[predictor_var] + control_var + ["covid19", "citation_per_month_log", "citation_per_month_binary_20perc", "citation_per_month_binary_10perc", "citation_per_month_binary_5perc", "citation_per_month_binary_1perc", "novelty_10perc"]]
    flu_matched = flu_matched[[predictor_var] + control_var + ["covid19", "citation_per_month_log", "citation_per_month_binary_20perc", "citation_per_month_binary_10perc", "citation_per_month_binary_5perc", "citation_per_month_binary_1perc", "novelty_10perc"]]
    combined_papers = pd.concat([covid19_matched, flu_matched])
    print("Number of matched samples: {}".format(combined_papers.shape[0]))

    # Standardize
    standardize_columns = [predictor_var] + ["new_tie_rate"] + ["topic_distr{}".format(i) for i in range(1,20)]
    X = np.array(combined_papers[standardize_columns])
    X_scaled = scale(X)

    combined_papers_scaled = combined_papers.copy()
    combined_papers_scaled[standardize_columns] = X_scaled

    ## Effect of topic familiarity
    binary_citation_20perc_tf_model = cem_test(0.2, "topic_familiarity", combined_papers_scaled, [predictor_var] + ["new_tie_rate"] + ["topic_distr{}".format(i) for i in range(1,20)])
    binary_citation_10perc_tf_model = cem_test(0.1, "topic_familiarity", combined_papers_scaled, [predictor_var] + ["new_tie_rate"] + ["topic_distr{}".format(i) for i in range(1,20)])
    binary_citation_5perc_tf_model = cem_test(0.05, "topic_familiarity", combined_papers_scaled, [predictor_var] + ["new_tie_rate"] + ["topic_distr{}".format(i) for i in range(1,20)])
    # binary_citation_1perc_tf_model = cem_test(0.01, "topic_familiarity", combined_papers_scaled, [predictor_var] + ["new_tie_rate"] + ["topic_distr{}".format(i) for i in range(1,20)])
    # export data for plotting.
    combined_papers_scaled.to_csv("../results/updated_results/cem_test_interaction_control_data_exported.csv", index=False)
    
    ## Effect of new collaboration tie rate
    # binary_citation_5perc_ntr_model = cem_test(0.05, "new_tie_rate", combined_papers_scaled, [predictor_var] + control_var)
