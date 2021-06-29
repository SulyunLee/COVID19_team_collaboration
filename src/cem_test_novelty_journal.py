


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import statsmodels.api as sm
import statsmodels.formula.api as smf
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

def cem_test(target_colname, var, combined_papers_scaled, feature_var):

    # Remove multi-collinearity
    aftervif = calculate_vif(combined_papers_scaled[feature_var])
    # aftervif = combined_papers_scaled[feature_var]

    features = [col.replace(col, 'C({})'.format(col)) if "publish_" in col else col for col in aftervif.columns]

    formula = "{} ~ C(covid19):{} + C(covid19) + {}".format(target_colname, var, "+".join(features))
    interaction_model = smf.ols(formula=formula, data=combined_papers_scaled).fit()
    write_filename = "{}_{}_cem_model_journal".format(target_colname, var)
    with open("../results/updated_results/{}.csv".format(write_filename), "w") as fh:
        fh.write(interaction_model.summary().as_csv())

    return interaction_model

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
    covid19_paper_df = covid19_paper_df.dropna(subset=[predictor_var] + control_var + ["novelty_10perc"])
    print("Number of COVID papers: {}".format(covid19_paper_df.shape[0]))

    # flu
    flu_paper_df = flu_paper_df.dropna(subset=[predictor_var] + control_var + ["novelty_10perc"])
    print("Number of flu papers: {}".format(flu_paper_df.shape[0]))

    print("Creating matching variable...")
    # match on journals
    covid19_journals = covid19_paper_df.journal.str.lower().str.strip()
    covid19_paper_df = covid19_paper_df.assign(journal = covid19_journals)

    flu_journals = flu_paper_df.publication_name.str.lower().str.strip()
    flu_paper_df = flu_paper_df.assign(journal = flu_journals)

    random.seed(100)

    matched_flu_scopusid = []
    matched_covid19_uid = []
    for idx, flu_paper in flu_paper_df.iterrows():
        flu_journal = flu_paper.journal
        matched_covid19_paper = covid19_paper_df[covid19_paper_df.journal == flu_journal]
        if matched_covid19_paper.shape[0] != 0:
            new_matched_covid19_paper = [p for p in matched_covid19_paper.cord_uid.values if p not in matched_covid19_uid]
            if len(new_matched_covid19_paper) != 0:
                sampled_covid19_match = random.choice(new_matched_covid19_paper)

                matched_flu_paper = flu_paper_df.iloc[idx].scopus_id
                matched_flu_scopusid.append(matched_flu_paper)
                matched_covid19_uid.append(sampled_covid19_match)

    print("Generating covid indicator variable...")
    covid19_matched = covid19_paper_df[covid19_paper_df.cord_uid.isin(matched_covid19_uid)]
    covid19_matched.reset_index(drop=True, inplace=True)
    covid19_matched = covid19_matched.assign(covid19=int(1))

    flu_matched = flu_paper_df[flu_paper_df.scopus_id.isin(matched_flu_scopusid)]
    flu_matched.reset_index(drop=True, inplace=True)
    flu_matched = flu_matched.assign(covid19=int(0))

    # Combine covid and flu papers
    covid19_matched = covid19_matched[[predictor_var] + control_var + ["covid19", "novelty_10perc"]]
    flu_matched = flu_matched[[predictor_var] + control_var + ["covid19", "novelty_10perc"]]
    combined_papers = pd.concat([covid19_matched, flu_matched])
    print("Number of matched samples: {}".format(combined_papers.shape[0]))

    # Standardize
    standardize_columns = [predictor_var] + ["avg_tdsim", "new_tie_rate", "hindex_gini","cultural_similarity", "topic_familiarity_var", "max_hindex_log", "team_size_log", "prac_affil_rate"] + ["topic_distr{}".format(i) for i in range(1,20)]
    X = np.array(combined_papers[standardize_columns])
    X_scaled = scale(X)

    combined_papers_scaled = combined_papers.copy()
    combined_papers_scaled[standardize_columns] = X_scaled

    ## Effect of topic familiarity
    novelty_tf_model = cem_test("novelty_10perc", "topic_familiarity", combined_papers_scaled, [predictor_var] + control_var)

    # export data for plotting
    combined_papers_scaled.to_csv("../results/updated_results/cem_test_interaction_journal_novelty_data_exported.csv", index=False)

