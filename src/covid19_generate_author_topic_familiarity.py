

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from generate_LDA import *
from compute_topic_distr_threshold import *
from sklearn.metrics.pairwise import cosine_similarity
from gensim.test.utils import datapath

def authorid_group_apply_func(group):
    group_papers = group.scopus_id.unique()
    group_topic_distr_df = unique_paper_abstracts.loc[unique_paper_abstracts['scopus_id'].isin(group_papers), 'topic_distr'].dropna().reset_index(drop=True)
    group_topic_distr_arr = np.array(group_topic_distr_df.tolist())

    group_avg_topic_distr = group_topic_distr_arr.mean(axis=0) # averaged topic distribution
    return group_avg_topic_distr

if __name__ == "__main__":
    print('Loading datasets...')
    covid19_paper_filename = "../dataset/COVID19_papers_modeling_final.csv"
    authorid_pub_database_filename = "../dataset/COVID19_authorid_pub_database.csv"
    covid19_paper_df = pd.read_csv(covid19_paper_filename)
    authorid_pub_df = pd.read_csv(authorid_pub_database_filename)

    infile = open("../dataset/scopusid_abstract_dict.pickle", "rb")
    scopusid_abstract_dict = pickle.load(infile)
    infile.close()

    print("Number of COVID-19 papers: {}".format(covid19_paper_df.shape[0]))

    author_id_eval = covid19_paper_df['author_id_modified'].apply(lambda x:eval(x))
    covid19_paper_df = covid19_paper_df.assign(author_id_modified=author_id_eval)

    # unique author IDs of COVID19 papers
    unique_authorid = np.unique(np.array([elem for singleList in covid19_paper_df.author_id_modified.tolist() for elem in singleList]))
    print('Number of unique authors in COVID19 papers...')
    print(unique_authorid.shape[0])

    # search authors' previous publications of the authors of COVID19 papers
    author_prev_pubs = authorid_pub_df[authorid_pub_df['author_id'].isin(unique_authorid)]
    print('Number of authors found from Scopus...')
    print(author_prev_pubs.author_id.unique().shape[0])
    author_prev_pubs.loc[:,'cover_date'] = pd.to_datetime(author_prev_pubs.loc[:,'cover_date'])

    # select authors' previous publications published before COVID-19.
    author_prev_pubs = author_prev_pubs[author_prev_pubs['cover_date'] < "2020-01-01"]
    author_prev_pubs = author_prev_pubs.reset_index(drop=True)
    print('Number of authors after excluding no papers before COVID19...')
    print(author_prev_pubs.author_id.unique().shape[0])
    print('Number of previous publications before COVID-19')
    print(author_prev_pubs.scopus_id.unique().shape[0])


    author_prev_pubs.loc[:,"abstract"] = author_prev_pubs.loc[:,"scopus_id"].map(scopusid_abstract_dict)

    # get authorid-scopusid matching retrievable from Scopus abstract API
    author_prev_pubs = author_prev_pubs[author_prev_pubs['abstract'] != 0].reset_index(drop=True)
    print('Number of authors after excluding those not found any abstracts...')
    print(author_prev_pubs.author_id.unique().shape[0])

    # unique papers and their abstracts
    unique_paper_abstracts = author_prev_pubs.drop_duplicates(subset=['scopus_id'])[["scopus_id", "title", "abstract"]].reset_index(drop=True)

    # replace nan abstracts with empty string
    abstract_replaced = unique_paper_abstracts['abstract'].replace(np.nan, '', regex=True)
    unique_paper_abstracts.loc[:,"abstract"] = abstract_replaced

    covid19_paper_abstracts = covid19_paper_df[["cord_uid", "title", "abstract"]]

    covid19_paper_prev_pub_combined = pd.concat([covid19_paper_abstracts, unique_paper_abstracts])

    # generate_LDA topic distribution of the title and abstract
    lda_model, bow_corpus, topic_num, lda_arr = generate_lda_topic_distr(covid19_paper_prev_pub_combined)


    # save model
    temp_file = datapath("covid19_paper_prev_pub_lda_model")
    lda_model.save(temp_file)

    # save topic distribution array
    outfile = open("temp/COVID19_paper_prev_pub_combined_lda_arr.pickle", "wb")
    pickle.dump(lda_arr, outfile)
    outfile.close()

    # save the separate topic distributions for focal paper and previous publications
    covid19_paper_combined_lda_arr = lda_arr[:covid19_paper_abstracts.shape[0],:]
    covid19_paper_combined_lda_dict = dict(zip(covid19_paper_abstracts.cord_uid, covid19_paper_combined_lda_arr))
    outfile = open("temp/COVID19_paper_combined_lda_dict.pickle", "wb")
    pickle.dump(covid19_paper_combined_lda_dict, outfile)
    outfile.close()

    prev_pub_combined_lda_arr = lda_arr[covid19_paper_abstracts.shape[0]:,:]
    prev_pub_combined_lda_dict = dict(zip(unique_paper_abstracts.scopus_id, prev_pub_combined_lda_arr))
    unique_paper_abstracts.loc[:,'topic_distr'] = pd.Series(prev_pub_combined_lda_arr.tolist())

    outfile = open("temp/COVID19_prev_pub_combined_lda_dict.pickle", "wb")
    pickle.dump(prev_pub_combined_lda_dict, outfile)
    outfile.close()

   # average of topic distributions for each author
    print('Computing average topic distributions for each author...')
    grouped = authorid_pub_df.groupby('author_id')
    author_arr = np.zeros((len(grouped))).astype(int)
    avg_topic_distr_arr = np.zeros((len(grouped))).astype(object)
    for idx, elem in tqdm(enumerate(grouped), total=len(grouped)):
        name, group = elem
        author_avg_topic_distr = authorid_group_apply_func(group)
        author_arr[idx] = name
        avg_topic_distr_arr[idx] = author_avg_topic_distr

    authorid_td_dict = dict(zip(author_arr, avg_topic_distr_arr))
    outfile = open("../dataset/COVID19_authorid_combined_lda_dict.pickle", "wb")
    pickle.dump(authorid_td_dict, outfile)
    outfile.close()


