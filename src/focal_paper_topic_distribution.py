
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from generate_LDA import *
from gensim.test.utils import datapath

if __name__ == "__main__":

    print('Loading datasets...')
    # covid19_paper_filename = "../dataset/COVID19_papers_modeling_final.csv"
    # covid19_paper_df = pd.read_csv(covid19_paper_filename)
    flu_paper_filename = "../dataset/FLU_papers_modeling_final.csv"
    flu_paper_df = pd.read_csv(flu_paper_filename)

    # generate LDA topic distribution of the title and abstracts
    print('Generating topic distributions...')
    lda_model, bow_corpus, topic_num, lda_arr = generate_lda_topic_distr(flu_paper_df) 

    temp_file = datapath("FLU_focal_paper_lda_model")
    lda_model.save(temp_file)

    # lda_model = LdaModel.load(temp_file)
    # outfile = open("temp/FLU_focal_paper_lda_model.pickle", "wb")
    # pickle.dump(lda_model, outfile)
    # outfile.close()
    
    outfile = open("temp/FLU_focal_paper_lda_arr.pickle", "wb")
    pickle.dump(lda_arr, outfile)
    outfile.close()

    # covid19_paper_df['topic_distr'] = pd.Series(lda_arr.tolist())
    # covid19_paper_df.to_csv(covid19_paper_filename, index=False, \
            # encoding='utf-8-sig')
    flu_paper_df['topic_distr'] = pd.Series(lda_arr.tolist())
    flu_paper_df.to_csv(flu_paper_filename, index=False, \
            encoding='utf-8-sig')
    



