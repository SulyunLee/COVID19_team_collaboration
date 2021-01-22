
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def plot_distribution(h_index_arr, outfile_name):
    sns.distplot(h_index_arr)
    plt.title("Distribution of h-index of authors from COVID-19 papers")
    plt.xlabel("h-index")
    plt.ylabel("Distribution")
    plt.xticks(np.arange(min(hindex_found), max(hindex_found)+1, 20))
    plt.savefig(outfile_name)
    plt.close()

def plot_author_perc(data):
    '''
    Plot the distribution of author percentages.
    The argument 'data' is the numpy array with percentages.
    '''
    print("plotting percentages...")
    # plot the distribution
    # the x-axis: the percentage of coauthors with abstracts
    # y-axis: the percentages of papers
    # sns.distplot(multiple_author_df['percent_authors_has_abstract'], kde=False, norm_hist=True)
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt.hist(data, bins, weights=np.ones(len(data))/len(data), cumulative=True,
            color="gray")
    plt.hist(data, bins, weights=np.ones(len(data))/len(data), cumulative=False)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title("Distribution of the percentages of coauthors with h-index found")
    plt.xlim(0,100)
    plt.xlabel("% of coauthors with h-index")
    plt.ylabel("%")
    plt.savefig("../plots/coauthor_perc_hindex_plt.png")
    plt.close()

if __name__ == "__main__":
    print('Loading datasets...')
    covid19_paper_filename = "../dataset/COVID19_papers_more.csv"
    paper_authorid_database_filename = "../dataset/paper_authorid_database_final.csv"
    authorid_hindex_dict_filename = "../dataset/authorid_hindex_before_covid19_dict.pickle"

    covid19_paper_df = pd.read_csv(covid19_paper_filename)
    paper_authorid_df = pd.read_csv(paper_authorid_database_filename)

    infile = open(authorid_hindex_dict_filename, "rb")
    authorid_hindex_dict = pickle.load(infile)
    infile.close()

    # match the COVID19 papers using paper-authorid database
    # extract only the papers with at least one authors are known
    covid19_paper_authorid_df = paper_authorid_df[paper_authorid_df.cord_uid.isin(covid19_paper_df.cord_uid.values)]
    covid19_paper_authorid_df.reset_index(drop=True, inplace=True)
    author_id_list = covid19_paper_authorid_df['author_id_modified'].apply(lambda x:eval(x))
    covid19_paper_authorid_df.insert(len(covid19_paper_authorid_df.columns), 
            'author_id_list', author_id_list)
    # select only the papers with one or more authors
    covid19_paper_authorid_df = covid19_paper_authorid_df[covid19_paper_authorid_df['author_id_modified'] != '[]']
    covid19_paper_authorid_df.reset_index(drop=True, inplace=True)
    print('Number of COVID-19 papers with one or more authors found from Scopus...')
    print(covid19_paper_authorid_df.shape[0])

    avg_hindex_arr = np.zeros((covid19_paper_authorid_df.shape[0])).astype(float)
    for idx, row in tqdm(covid19_paper_authorid_df.iterrows(), total=covid19_paper_authorid_df.shape[0]):
    # for idx, row in tqdm(covid19_paper_authorid_df.iloc[:5].iterrows(), total=covid19_paper_authorid_df.shape[0]):
        author_lst = row.author_id_list
        hindex_lst = []
        for author_id in author_lst:
            # if author has papers before COVID-19
            if int(author_id) in authorid_hindex_dict:
                # find the h-index of a corresponding author
                h_index = authorid_hindex_dict[int(author_id)]
            else:
                # if the author does not have papers before COVID-19,
                # h-index is zero
                h_index = 0
            hindex_lst.append(h_index)
        # remove h_index not found from Scopus
        hindex_found_lst = [x for x in hindex_lst if x != -1]
        avg_hindex = sum(hindex_found_lst) / len(hindex_found_lst)
        avg_hindex_arr[idx] = avg_hindex

    # save the dictionary
    # Key: covid19 cord_uid, value: average h-index of coauthors
    covid19_avg_hindex_dict = dict(zip(covid19_paper_authorid_df.cord_uid.values, avg_hindex_arr))
    outfile = open("../dataset/covid19_avg_hindex_dict.pickle", "wb")
    pickle.dump(covid19_avg_hindex_dict, outfile)
    outfile.close()
    









     

