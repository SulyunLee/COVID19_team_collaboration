'''
This script calculates the h-index of authors
considering the publications before Influenza(01-01-2019)
'''

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from pyscopus import Scopus
import requests

def hIndex(citations):
    """
    citations: list[int]
    """
    if not citations:
        return 0
    citations.sort()
    h_index = 0
    size = len(citations)
    for i in range(size):
        try_index = size - i
        if citations[i] >= try_index and (i < 1 or citations[i - 1] <= try_index):
            h_index = max(h_index , try_index)

    return h_index

if __name__ == "__main__":
    print('Loading datasets...')
    authorid_pub_database_filename = "../dataset/FLU_authorid_pub_database.csv"
    authorid_pub_df = pd.read_csv(authorid_pub_database_filename)

    # select papers before flu
    authorid_pub_df = authorid_pub_df[authorid_pub_df['cover_date'] < "2019-01-01"]

    # select only needed columns
    authorid_pub_df = authorid_pub_df[["author_id", "cover_date", "citation_count"]]

    # group by author IDs and create lists with citation counts of their 
    # publications before COVID-19
    citation_by_authorid = authorid_pub_df.groupby('author_id')['citation_count'].apply(list)
    # calculate h-index based on citations before flu
    hindex_before_flu_by_authorid = citation_by_authorid.apply(lambda x:hIndex(x))

    # Create dictionary
    # Key: Author ID
    # Value: calculated h-index before flu
    authorid_hindex_dict = hindex_before_flu_by_authorid.to_dict()

    # save the dictionary
    outfile = open("../dataset/authorid_hindex_before_flu_dict.pickle", "wb")
    pickle.dump(authorid_hindex_dict, outfile)
    outfile.close()


