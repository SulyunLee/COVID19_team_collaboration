
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
    flu_paper_filename = "../dataset/FLU_papers_modeling_final.csv"
    authorid_pub_database_filename = "../dataset/FLU_authorid_pub_database.csv"
    flu_paper_df = pd.read_csv(flu_paper_filename)
    authorid_pub_df = pd.read_csv(authorid_pub_database_filename)

    # only select previous 10 years of publications
    authorid_pub_df = authorid_pub_df[(authorid_pub_df["cover_date"] > "2009-01-01")&(authorid_pub_df["cover_date"] < "2019-01-01")]
    authorid_pub_df.reset_index(drop=True, inplace=True)

    # import abstracts of publications published before 2019 for 10 years
    # Dictionary-Key:Scopus ID, Value: Abstract
    infile = open("../dataset/flu_scopusid_abstract_dict.pickle", "rb")
    scopusid_abstract_dict = pickle.load(infile)
    infile.close()

    print("Number of flu papers: {}".format(flu_paper_df.shape[0]))

    author_id_eval = flu_paper_df['authors'].apply(lambda x:eval(x))
    flu_paper_df = flu_paper_df.assign(authors=author_id_eval)

    # unique author IDs of COVID19 papers
    unique_authorid = np.unique(np.array([elem for singleList in flu_paper_df.authors.tolist() for elem in singleList]))
    print('Number of unique authors in COVID19 papers...')
    print(unique_authorid.shape[0])

    author_prev_pubs = authorid_pub_df[authorid_pub_df['author_id'].isin(unique_authorid)]
    print('Number of authors found from Scopus...')
    print(author_prev_pubs.author_id.unique().shape[0])

    print('Number of authors after excluding no papers before 2019...')
    print(authorid_pub_df.author_id.unique().shape[0])
    print("Number of previous publications before 2019")
    print(authorid_pub_df.scopus_id.unique().shape[0])

    # Map the abstract with Scopus ID
    authorid_pub_df.loc[:,"abstract"] = authorid_pub_df.loc[:,"scopus_id"].map(scopusid_abstract_dict)

    # get authorid-scopusid matching retrievable from Scopus abstract API
    authorid_pub_df = authorid_pub_df[authorid_pub_df['abstract'] != 0].reset_index(drop=True)
    print('Number of authors after excluding those not found any abstracts...')
    print(authorid_pub_df.author_id.unique().shape[0])

    # unique papers and their abstracts
    unique_paper_abstracts = authorid_pub_df.drop_duplicates(subset=['scopus_id'])[["scopus_id", "title", "abstract"]].reset_index(drop=True)

    # replace nan abstracts with empty string
    abstract_replaced = unique_paper_abstracts['abstract'].replace(np.nan, '', regex=True)
    unique_paper_abstracts.loc[:,"abstract"] = abstract_replaced

    flu_paper_abstracts = flu_paper_df[["scopus_id", "title", "abstract"]]

    flu_paper_prev_pub_combined = pd.concat([flu_paper_abstracts, unique_paper_abstracts])


    # generate LDA topic distribution of the title and abstract
    lda_model, bow_corpus, topic_num, lda_arr = generate_lda_topic_distr(flu_paper_prev_pub_combined) 

    # save model
    temp_file = datapath("flu_paper_prev_pub_lda_model")
    lda_model.save(temp_file)

    # save topic distribution array
    outfile = open("temp/FLU_paper_prev_pub_combined_lda_arr.pickle", "wb")
    pickle.dump(lda_arr, outfile)
    outfile.close()

    # save the separate topic distributions for focal paper and previous publications
    flu_paper_combined_lda_arr = lda_arr[:flu_paper_abstracts.shape[0],:]
    flu_paper_combined_lda_dict = dict(zip(flu_paper_abstracts.scopus_id, flu_paper_combined_lda_arr))
    outfile = open("temp/FLU_paper_combined_lda_dict.pickle", "wb")
    pickle.dump(flu_paper_combined_lda_dict, outfile)
    outfile.close()

    prev_pub_combined_lda_arr = lda_arr[flu_paper_abstracts.shape[0]:,:]
    prev_pub_combined_lda_dict = dict(zip(unique_paper_abstracts.scopus_id, prev_pub_combined_lda_arr))
    unique_paper_abstracts.loc[:,'topic_distr'] = pd.Series(prev_pub_combined_lda_arr.tolist())

    outfile = open("temp/FLU_prev_pub_combined_lda_dict.pickle", "wb")
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
    outfile = open("../dataset/FLU_authorid_combined_lda_dict.pickle", "wb")
    pickle.dump(authorid_td_dict, outfile)
    outfile.close()


    
