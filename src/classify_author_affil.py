from pyscopus import *
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import requests

def gather_missing_authorid_affil(authorid_arr, authorid_affil_df, save_filename):
    # get the authors' affiliations if not exist in the database
    scopus = Scopus(key)
    for authorid in tqdm(authorid_arr):
        if int(authorid) not in authorid_affil_df.author_id.values:
            try:
                author_df = scopus.search_author("AU-ID({})".format(authorid))
                authorid_affil_df = pd.concat([authorid_affil_df, author_df], axis=0)
            except KeyError:
                print("Author ID {} not found".format(authorid))

    authorid_affil_df.to_csv("../dataset/{}".format(save_filename), index=False, encoding='utf-8-sig')

    return(authorid_affil_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-key', '--key', type=str)

    args = parser.parse_args()
    key = args.key

    print('Loading datasets...')
    covid19_paper_filename = "../dataset/COVID19_papers_modeling.csv"
    paper_authorid_database_filename = "../dataset/paper_authorid_database_final.csv"
    authorid_affil_database_filename = "../dataset/authorid_affil_database_final.csv"
    authorid_info_database_filename = "../dataset/covid19_authorid_info_database.csv"

    covid19_paper_df = pd.read_csv(covid19_paper_filename)
    paper_authorid_df = pd.read_csv(paper_authorid_database_filename)
    authorid_affil_df = pd.read_csv(authorid_affil_database_filename)
    authorid_info_df = pd.read_csv(authorid_info_database_filename)
    print(authorid_affil_df.shape[0])

    # select only the paper-authorid matching of COVID19 papers
    covid19_paper_authorid_df = paper_authorid_df[paper_authorid_df.cord_uid.isin(covid19_paper_df.cord_uid.values)]
    covid19_paper_authorid_df = covid19_paper_authorid_df.reset_index(drop=True)
    covid19_paper_authorid_df['author_id_modified'] = covid19_paper_authorid_df['author_id_modified'].apply(lambda x:eval(x))

    author_id_lists = covid19_paper_authorid_df.author_id_modified.values
    # unique author IDs of COVID-19 papers
    unique_authorid = np.unique(np.array([elem for singleList in author_id_lists.tolist() for elem in singleList]))

    #############################################
    # get the authors' affiliations if not exist in the database
    # authorid_affil_df = gather_missing_authorid_affil(unique_authorid, authorid_affil_df, "authorid_affil_database_final.csv")
    #############################################

    # classify if the autors' affiliations are practical centers
    keywords = ["hospital", "clinic", "national center", "ministry of", "medical center"]
    affiliation_name_lower = authorid_affil_df.affiliation.str.lower()
    practical_affil = affiliation_name_lower.str.contains('|'.join(keywords))
    authorid_affil_df['practical_affil'] = practical_affil

    authorid_affil_df.to_csv("../dataset/authorid_affil_database_final.csv", index=False, encoding="utf-8-sig")






