import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns

def compute_author_percentage(df, all_author_colname, author_with_abstract_colname):
    '''
    This function computes the percentages of authors available with their previous
    publication abstracts (from Scopus) among all the available authors.
    all_author_colname: column name with the lists of all authors
    author_with_abstract_colname: column name with the lists of available authors with abstracts.
    '''

    df['num_all_author'] = df[all_author_colname].apply(lambda x:len(x))
    df['num_author_with_abs'] = df[author_with_abstract_colname].apply(lambda x:len(x))
    multiple_author_df = df[df['num_all_author'] > 1]
    print('The number of COVID19 papers with more than 1 authors')
    print(multiple_author_df.shape[0])

    # compute the percentage of coauthors with abstracts
    print("Computing percentages...")
    multiple_author_df.loc[:,'percent_authors_has_abstract'] = multiple_author_df['num_author_with_abs'] / multiple_author_df['num_all_author'] * 100

    return multiple_author_df



def plot_author_perc(data):
    '''
    Plot the distribution of author percentages.
    The argument 'data' is the numpy array with percentages.
    '''
    print("plotting...")
    # plot the distribution
    # the x-axis: the percentage of coauthors with abstracts
    # y-axis: the percentages of papers
    # sns.distplot(multiple_author_df['percent_authors_has_abstract'], kde=False, norm_hist=True)
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt.hist(data, bins, weights=np.ones(len(data))/len(data), cumulative=True,
            color="gray")
    plt.hist(data, bins, weights=np.ones(len(data))/len(data), cumulative=False)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title("Distribution of the percentages of coauthors with abstracts")
    plt.xlim(0,100)
    plt.xlabel("% of coauthors with abstracts")
    plt.ylabel("%")
    plt.savefig("../plots/coauthor_perc_plt.png")
    plt.close()


def papers_with_threshold(threshold, multiple_author_df):
    above_threshold_paperid = multiple_author_df.loc[multiple_author_df['percent_authors_has_abstract'] >= threshold*100,'cord_uid'].values

    return above_threshold_paperid





