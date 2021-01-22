

import pandas as pd
import numpy as np
import random
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

def generate_binary_citation_var(perc, df):
    cut_point = int(df.shape[0]*perc) 
    cut = df.citation_count_log.sort_values().iloc[-cut_point] 
    citation_counts_binary = np.where(df.citation_count_log > cut,1,0)
    var_name = "citation_binary_{}perc".format(int(perc*100))
    df = df.assign(**{var_name:citation_counts_binary})

    return df

def cem_test(perc, var, combined_papers_scaled, feature_var):
    # remove multi-collinearity
    aftervif = calculate_vif(combined_papers_scaled[[x for x in feature_var if x != var]])

    formula = "citation_binary_{}perc ~ covid19:{} + {}".format(int(perc*100), var, "+".join(aftervif.columns))
    model = smf.ols(formula=formula, data=combined_papers_scaled).fit()

    write_filename = "citation_binary_{}perc_{}_cem_model".format(int(perc*100), var)
    with open("../results/{}.csv".format(write_filename), "w") as fh:
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
    covid19_paper_df = covid19_paper_df.assign(team_size_log = team_size_log_transformed)
    covid19_paper_df = covid19_paper_df.assign(max_hindex_log = max_hindex_log_transformed)
    covid19_paper_df = covid19_paper_df.assign(citation_count_log = citation_count_log_transformed)

    # flu
    team_size_log_transformed = np.log(flu_paper_df['team_size']+1)
    max_hindex_log_transformed = np.log(flu_paper_df['max_hindex']+1)
    citation_count_log_transformed = np.log(flu_paper_df['citation_count']+1)

    flu_paper_df = flu_paper_df.assign(team_size_log = team_size_log_transformed)
    flu_paper_df = flu_paper_df.assign(max_hindex_log = max_hindex_log_transformed)
    flu_paper_df = flu_paper_df.assign(citation_count_log = citation_count_log_transformed)

    feature_var = ["avg_tdsim", "new_tie_rate", "hindex_gini","cultural_similarity", "topic_familiarity", "max_hindex_log", "time_since_pub", "team_size_log", "prac_affil_rate"] + ["topic_distr{}".format(i) for i in range(1,20)]

    # drop NA rows
    # COVID19
    covid19_paper_df = covid19_paper_df.dropna(subset=feature_var + ["citation_count_log"])
    print("Number of COVID papers: {}".format(covid19_paper_df.shape[0]))

    # flu
    flu_paper_df = flu_paper_df.dropna(subset=feature_var + ["citation_count_log"])
    print("Number of flu papers: {}".format(flu_paper_df.shape[0]))

    print("Creating matching variable...")
    team_size_series = pd.concat([covid19_paper_df.team_size, flu_paper_df.team_size])
    avg_team_size = team_size_series.mean()
    covid19_team_size_binary = covid19_paper_df.team_size.apply(lambda x: 1 if x > avg_team_size else 0)
    covid19_paper_df = covid19_paper_df.assign(team_size_binary=covid19_team_size_binary)

    flu_team_size_binary = flu_paper_df.team_size.apply(lambda x: 1 if x > avg_team_size else 0)
    flu_paper_df = flu_paper_df.assign(team_size_binary=flu_team_size_binary)

    covid19_matched0 = covid19_paper_df[covid19_paper_df.team_size_binary == 0].sample(n=1240)
    covid19_matched1 = covid19_paper_df[covid19_paper_df.team_size_binary == 1].sample(n=944)
    covid19_matched = pd.concat([covid19_matched0, covid19_matched1])

    print("Generating covid indicator variable...")
    covid19_matched = covid19_matched.assign(covid19=1)
    flu_paper_df = flu_paper_df.assign(covid19=0)

    # Generate binary citation variable
    # Combine covid and flu papers
    covid19_matched = covid19_matched[feature_var + ["covid19", "citation_count_log"]]
    covid19_matched = generate_binary_citation_var(0.2, covid19_matched)
    covid19_matched = generate_binary_citation_var(0.1, covid19_matched)
    covid19_matched = generate_binary_citation_var(0.05, covid19_matched)

    flu_matched = flu_paper_df[feature_var + ["covid19", "citation_count_log"]]
    flu_matched = generate_binary_citation_var(0.2, flu_matched)
    flu_matched = generate_binary_citation_var(0.1, flu_matched)
    flu_matched = generate_binary_citation_var(0.05, flu_matched)
    combined_papers = pd.concat([covid19_matched, flu_matched])

    # Standardize
    X = np.array(combined_papers[feature_var])
    X_scaled = scale(X)

    combined_papers_scaled = combined_papers.copy()
    combined_papers_scaled[feature_var] = X_scaled

    ## Effect of topic familiarity
    binary_citation_tf_model = cem_test(0.2, "topic_familiarity", combined_papers_scaled, feature_var)
    binary_citation_ntr_model = cem_test(0.2, "new_tie_rate", combined_papers_scaled, feature_var)
    binary_citation_tf_model = cem_test(0.1, "topic_familiarity", combined_papers_scaled, feature_var)
    binary_citation_ntr_model = cem_test(0.1, "new_tie_rate", combined_papers_scaled, feature_var)
    binary_citation_tf_model = cem_test(0.05, "topic_familiarity", combined_papers_scaled, feature_var)
    binary_citation_ntr_model = cem_test(0.05, "new_tie_rate", combined_papers_scaled, feature_var)