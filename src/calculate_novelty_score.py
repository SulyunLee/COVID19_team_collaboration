
import pandas as pd
import numpy as np
import pickle
import json
from tqdm import tqdm
from iso4 import abbreviate
import itertools

def convert_journal_id(journal_list, journal_dict):
    journal_id_list = []
    for journal in journal_list:
        if journal not in journal_dict:
            journal_id_list.append(-1)
        elif journal != '':
            journal_id = journal_dict[journal]
            journal_id_list.append(journal_id)

        

    return journal_id_list


if __name__ == "__main__":
    universe_paper_filename = "../dataset/all_published_papers_year2019.csv"
    focal_paper_filename = "../dataset/FLU_papers_modeling_with_features.csv"
    focal_paper_ref_filename = "temp/journal_retrieval/FLU_papers_journal_list_arr1.npz"

    print("Loading data...")
    universe_paper_df = pd.read_csv(universe_paper_filename)
    focal_paper_df = pd.read_csv(focal_paper_filename)
    data = np.load(focal_paper_ref_filename)
    focal_paper_ref = data['arr_0']
    data.close()

    # convert list of journals to list using eval
    journal_list_eval = universe_paper_df['journal_list'].apply(lambda x: eval(x))
    universe_paper_df = universe_paper_df.assign(journal_list=journal_list_eval)

    # exclude papers with no reference info
    universe_paper_ref_df = universe_paper_df[universe_paper_df.journal_list != 0]
    universe_paper_ref_df.reset_index(drop=True, inplace=True)

    # generate dictionary:
    # Key: journal name, Value: integer ID
    print("Generating journal name-ID matching dictionary...")
    l = universe_paper_ref_df.journal_list.values.tolist()
    merged_l = list(itertools.chain(*l))
    journal_string_list = list(set(merged_l))
    journal_string_list = journal_string_list
    journal_id_list = list(range(0, len(journal_string_list)))
    journal_dict = dict(zip(journal_string_list, journal_id_list))

    # match journal id
    print("Matching to journal IDs")
    journal_id_arr = np.zeros((universe_paper_ref_df.shape[0])).astype(object)
    for idx, row in tqdm(universe_paper_ref_df.iterrows(), total=universe_paper_ref_df.shape[0]):
        journal_list = row.journal_list
        journal_id_list = convert_journal_id(journal_list, journal_dict)
        journal_id_arr[idx] = journal_id_list
        
    universe_paper_ref_df = universe_paper_ref_df.assign(journal_id_list=journal_id_arr)

    print("Generating universe paper journal pair dictionary...")
    ########################################
    journal_pair_dict = {}
    for idx, row in tqdm(universe_paper_ref_df.iterrows(), total=universe_paper_ref_df.shape[0]):
        journal_list = row.journal_id_list
        journal_id_pairs = list(itertools.combinations(journal_list, 2))
        for pair in journal_id_pairs:
            if pair in journal_pair_dict:
                journal_pair_dict[pair] += 1
            elif (pair[1],pair[0]) in journal_pair_dict:
                journal_pair_dict[(pair[1],pair[0])] += 1
            else:
                journal_pair_dict[pair] = 1

    print("The number of journal pairs: {}".format(len(journal_pair_dict)))
    # journal_pair_data = pd.DataFrame.from_dict(journal_pair_dict, orient='index', columns=['count'])
    # journal_pair_data.to_csv("temp/2019_papers_journal_pair_count.csv")

    # outfile = open("temp/2019_papers_journal_pair_dict.pickle", "wb")
    # pickle.dump(journal_pair_dict, outfile)
    # outfile.close()

    #######################################
    # infile = open("temp/2019_papers_journal_pair_dict.pickle", "rb")
    # journal_pair_dict = pickle.load(infile)
    # infile.close()
    print("Generating journal including pair dictionary...")
    # Key: journal ID, Value: the number of pairs including the journal
    journal_include_dict = {}
    for pair in tqdm(journal_pair_dict.keys(), total=len(journal_pair_dict)):
        if pair[0] in journal_include_dict:
            journal_include_dict[pair[0]] += 1
        else:
            journal_include_dict[pair[0]] = 1

        if pair[1] in journal_include_dict:
            journal_include_dict[pair[1]] += 1
        else:
            journal_include_dict[pair[1]] = 1



    print("Commonness of journal pairs")
    commonness_dict = {}
    num_all_pairs = sum(journal_pair_dict.values())

    for pair in tqdm(journal_pair_dict.keys(), total=len(journal_pair_dict)):
        try:
            pair_count = journal_pair_dict[pair]
        except:
            pair_count = journal_pair_dict[(pair[1],pair[0])]

        commonness = (pair_count * num_all_pairs) / (journal_include_dict[pair[0]] * journal_include_dict[pair[1]])
        commonness_dict[pair] = commonness

    del journal_include_dict
    del journal_pair_dict
    del universe_paper_ref_df

    print("Matching focal paper journal IDs")
    # journal not in the universe: -1
    focal_journal_id_arr = np.zeros((focal_paper_ref.shape[0])).astype(object)
    for idx, journal_list in tqdm(enumerate(focal_paper_ref), total=focal_paper_ref.shape[0]):
        if journal_list != 0:
            journal_id_list = convert_journal_id(journal_list, journal_dict)
            focal_journal_id_arr[idx] = journal_id_list

    print("Generating focal paper reference journal pairs...")
    focal_journal_pair_arr = np.zeros((focal_journal_id_arr.shape[0])).astype(object)
    for idx, journal_id_list in tqdm(enumerate(focal_journal_id_arr), total=focal_journal_id_arr.shape[0]):
        if journal_id_list == 0: # journal info not found
            focal_journal_pair_arr[idx] = 0
        else:
            focal_journal_id_pairs = list(itertools.combinations(journal_id_list,2))
            focal_journal_pair_arr[idx] = focal_journal_id_pairs

    del focal_journal_id_arr
    del focal_paper_ref


    print("Generating focal papaer journal pair commonness...")
    focal_commonness_arr = np.zeros((focal_journal_pair_arr.shape[0])).astype(object)
    for idx, journal_pairs in tqdm(enumerate(focal_journal_pair_arr), total=focal_journal_pair_arr.shape[0]):
        focal_commonness = []
        if journal_pairs == 0:
            focal_commonness.append(0)
        else:
            for pair in journal_pairs:
                if (pair[0],pair[1]) in commonness_dict:
                    focal_commonness.append(commonness_dict[(pair[0],pair[1])])
                elif (pair[1],pair[0]) in commonness_dict:
                    focal_commonness.append(commonness_dict[(pair[1],pair[0])])
                else:
                    focal_commonness.append(0)

        focal_commonness_arr[idx] = focal_commonness

    focal_paper_df = focal_paper_df.assign(commonness_list=focal_commonness_arr)
    focal_paper_df.to_csv(focal_paper_filename, index=False)









    




