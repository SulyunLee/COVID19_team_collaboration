import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from generate_LDA import *
from compute_topic_distr_threshold import *
from sklearn.metrics.pairwise import cosine_similarity

def authorid_group_apply_func(group):
    group_papers = group.scopus_id.unique()
    group_topic_distr_df = unique_paper_abstracts.loc[unique_paper_abstracts['scopus_id'].isin(group_papers), 'topic_distr'].dropna().reset_index(drop=True)
    group_topic_distr_arr = np.array(group_topic_distr_df.tolist())

    group_avg_topic_distr = group_topic_distr_arr.mean(axis=0) # averaged topic distribution
    return group_avg_topic_distr

def compute_avg_tdsim_from_threshold(threshold, multiple_author_df):
    above_threshold_paperid = papers_with_threshold(threshold, multiple_author_df)
    print('Papers above threshold {}: {}'.format(threshold, above_threshold_paperid.shape[0]))

    # compute the diversity of authors in COVID-19 papers
    print('Computing average cosine similarity for authors in papers with threshold...')
    covid19_paper_above_threshold = covid19_paper_df[covid19_paper_df.cord_uid.isin(above_threshold_paperid)].reset_index(drop=True)

    avg_pairwise_sim_arr = np.zeros((covid19_paper_above_threshold.shape[0])).astype(float)
    
    #iterate over all COVID-19 papers to compute pairwise topic distribution similarity
    for idx, row in tqdm(covid19_paper_above_threshold.iterrows(), total=covid19_paper_above_threshold.shape[0]):
    # for idx, row in tqdm(covid19_paper_authorid_df.iloc[:10].iterrows(), total=covid19_paper_authorid_df.shape[0]):
        
        authors_with_ab = row.author_id_with_ab
        num_authors = len(authors_with_ab)
        if num_authors == 1 or num_authors == 0:
            avg_pairwise_sim_arr[idx] = -1
        else:
            mask = np.isin(author_arr, authors_with_ab)
            author_arr_idx = np.argwhere(mask).T[0]
            author_avg_topic_distr = avg_topic_distr_arr[author_arr_idx]
            author_avg_topic_distr = author_avg_topic_distr[~pd.isnull(author_avg_topic_distr)]
            author_avg_topic_distr = np.stack(author_avg_topic_distr, axis=0) # convert to 2d numpy array

            # compute pair-wise cosine similarity
            pairwise_topic_distr_cs = cosine_similarity(author_avg_topic_distr)
            # compute averaged pairwise cosine similarity of the topic distributions
            iu = np.triu_indices(pairwise_topic_distr_cs.shape[0], 1)
            all_pairs_similarity = pairwise_topic_distr_cs[iu]
            average_similarity = all_pairs_similarity.sum() / all_pairs_similarity.shape[0]
            avg_pairwise_sim_arr[idx] = average_similarity

    outfile = open("temp/avg_pairwise_td_sim_threshold{}.pickle".format(threshold), "wb")
    pickle.dump(avg_pairwise_sim_arr, outfile)
    outfile.close()

    return avg_pairwise_sim_arr


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

    #####################################################
    # generate LDA topic distribution of the title and abstract
    lda_model, bow_corpus, topic_num, lda_arr = generate_lda_topic_distr(unique_paper_abstracts) 

    outfile = open("temp/covid19_prev_pub_lda_arr.pickle", "wb")
    pickle.dump(lda_arr, outfile)
    outfile.close()
    
    # print('Loading LDA results...')
    # infile = open("temp/covid19_prev_pub_lda_arr.pickle", "rb")
    # lda_arr = pickle.load(infile)
    # infile.close()

    unique_paper_abstracts.loc[:,'topic_distr'] = pd.Series(lda_arr.tolist())

   # average of topic distributions for each author
    print('Computing average topic distributions for each author...')
    grouped = author_prev_pubs.groupby('author_id')     
    author_arr = np.zeros((len(grouped))).astype(int)
    avg_topic_distr_arr = np.zeros((len(grouped))).astype(object)
    for idx, elem in tqdm(enumerate(grouped), total=len(grouped)):
        name, group = elem
        author_avg_topic_distr = authorid_group_apply_func(group)
        author_arr[idx] = name
        avg_topic_distr_arr[idx] = author_avg_topic_distr

    authorid_td_dict = dict(zip(author_arr, avg_topic_distr_arr))
    outfile = open("../dataset/covid19_authorid_td_dict.pickle", "wb")
    pickle.dump(authorid_td_dict, outfile)
    outfile.close()

    # find the authors with the abstracts available.
    print('Finding authors with abstracts available...')
    authors_with_ab_arr = np.zeros((covid19_paper_df.shape[0])).astype(object)
    for idx, row in tqdm(covid19_paper_df.iterrows(), total=covid19_paper_df.shape[0]):
        authors = row.author_id_modified
        authors = [int(author) for author in authors]

        mask = np.isin(author_arr, authors)
        author_arr_idx = np.argwhere(mask).T[0]
        authors_with_abstract = list(author_arr[author_arr_idx])
        authors_with_ab_arr[idx] = authors_with_abstract

    covid19_paper_df.loc[:,'author_id_with_ab'] = authors_with_ab_arr

    ################## THRESHOLD EXPERIMENTS ######################
    # # set the threshold and get the papers above the threshold
    # multiple_author_df = compute_author_percentage(covid19_paper_authorid_df, "author_id_modified", "author_id_with_ab")
    # plot_author_perc(multiple_author_df.percent_authors_has_abstract.values)
    # avg_pairwise_sim_arr = compute_avg_tdsim_from_threshold(0, multiple_author_df)
    # print("Number of papers with threshold: {}".format(avg_pairwise_sim_arr.shape[0]))
    # print(avg_pairwise_sim_arr.mean())

    # plot_author_perc(multiple_author_df.percent_authors_has_abstract.values)
    # avg_pairwise_sim_arr = compute_avg_tdsim_from_threshold(0.25, multiple_author_df)
    # print("Number of papers with threshold: {}".format(avg_pairwise_sim_arr.shape[0]))
    # print(avg_pairwise_sim_arr.mean())

    # avg_pairwise_sim_arr = compute_avg_tdsim_from_threshold(0.5, multiple_author_df)
    # print("Number of papers with threshold: {}".format(avg_pairwise_sim_arr.shape[0]))
    # print(avg_pairwise_sim_arr.mean())

    # avg_pairwise_sim_arr = compute_avg_tdsim_from_threshold(0.75, multiple_author_df)
    # print("Number of papers with threshold: {}".format(avg_pairwise_sim_arr.shape[0]))
    # print(avg_pairwise_sim_arr.mean())

    ###############################################################

    multiple_author_df = compute_author_percentage(covid19_paper_df, "author_id_modified", "author_id_with_ab")
    avg_pairwise_sim_arr = compute_avg_tdsim_from_threshold(0, multiple_author_df)
    print(avg_pairwise_sim_arr.mean())

    # append the average pairwise topic similarity to COVID-19 paper dataframe
    multiple_author_df = multiple_author_df.assign(avg_pairwise_td_sim=avg_pairwise_sim_arr)

    # write dictionary
    # key: cord_uid, value: avg_pairwise_similarity
    # only includes COVID-19 papers with more than one authors
    # Among papers with at least two authors, if only zero or one author has
    # abstracts available, the values are -1
    covid19_avg_tdsim_dict = dict(zip(multiple_author_df.cord_uid, multiple_author_df.avg_pairwise_td_sim))

    outfile = open("../dataset/covid19_avg_tdsim_dict.pickle", "wb")
    pickle.dump(covid19_avg_tdsim_dict, outfile)
    outfile.close()



















   
