import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import requests
from namsorclient import NamsorClient
from namsorclient.country_codes import CountryCodes
from namsorclient.request_objects import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-startidx', '--startidx', type=int)
    parser.add_argument('-endidx', '--endidx', type=int)
    parser.add_argument('-filenum', '--filenum', type=int)
    args = parser.parse_args()
    start_idx = args.startidx
    end_idx = args.endidx
    filenum = args.filenum

    # API key
    # key = "84cf9140e7d1cd12c07755eab3bee72f"

    print('Loading datasets...')
    covid19_paper_filename = "../dataset/COVID19_papers_modeling.csv"
    paper_authorid_database_filename = "../dataset/paper_authorid_database_final.csv"
    authorid_info_database_filename = "../dataset/covid19_authorid_info_database.csv"
    diaspora_data_filename = "../dataset/diaspora.csv"

    covid19_paper_df = pd.read_csv(covid19_paper_filename)
    paper_authorid_df = pd.read_csv(paper_authorid_database_filename)
    authorid_info_df = pd.read_csv(authorid_info_database_filename)
    diaspora_df = pd.read_csv(diaspora_data_filename, sep='|')

    # select only the paper-authorid matching of COVID19 papers
    covid19_paper_authorid_df = paper_authorid_df[paper_authorid_df.cord_uid.isin(covid19_paper_df.cord_uid.values)]
    covid19_paper_authorid_df = covid19_paper_authorid_df.reset_index(drop=True)
    covid19_paper_authorid_df['author_id_modified'] = covid19_paper_authorid_df['author_id_modified'].apply(lambda x:eval(x))

    author_id_lists = covid19_paper_authorid_df.author_id_modified.values
    # unique author IDs of COVID-19 papers
    unique_authorid = np.unique(np.array([elem for singleList in author_id_lists.tolist() for elem in singleList]))
    print("Number of unique authors in COVID-19 papers: {}".format(unique_authorid.shape[0]))

    diaspora_ethnicity_df = diaspora_df[["firstName","lastName","ethnicity"]]


    # add the country column to the author IDs
    authorid_info_ethnicity_df = pd.merge(authorid_info_df, diaspora_ethnicity_df, how="left",\
            left_on=["first","last"], right_on=["firstName","lastName"])

    not_found = 0
    for authorid in tqdm(unique_authorid):
        try:
            eth = authorid_info_ethnicity_df[authorid_info_ethnicity_df['author-id'] == int(authorid)]
            ethnicity = eth.ethnicity
            if ethnicity.isnull().sum() == 1:
                not_found += 1
        except:
            continue


    # instance of NamsorClient
    # client = NamsorClient(key)

    # iterate over author IDs in authorid_info database
    # country_arr = np.zeros((authorid_info_df.shape[0])).astype(str)
    # for idx, row in tqdm(authorid_info_df.iloc[:20].iterrows(), total=authorid_info_df.shape[0]):
    # for idx, row in tqdm(authorid_info_df.iloc[start_idx:end_idx].iterrows(), total=authorid_info_df.shape[0]):
        # first_name = row['first']
        # last_name = row['last']

        # response = client.origin(first_name=first_name, last_name=last_name)
        # origin_country = response.country_origin
        # country_arr[idx] = origin_country

    # outfile = open("temp/origin_country_arr{}.pickle".format(filenum), "wb")
    # pickle.dump(country_arr, outfile)
    # outfile.close()

    # authorid_info_df['origin_country'] = country_arr
    # authorid_info_df.to_csv("../dataset/covid19_authorid_info_database_final.csv", index=False)








