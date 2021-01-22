import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import combinations
import pickle

def construct_covid19_network(df, author_list_colname):
    '''
    Input:
      - df: dataframe that includes unique COVID-19 papers
            and their coauthors (list form)
      - author_list_colname: the name of the column that includes
            the lists of coauthors

    Output:
      - g: networkx graph of paper collaborations
           Nodes are the authors who participated in the papers and
           the edges are the coauthorship
    '''
    
    print("Generating nodes...") 
    # get unique author IDs of participated in the papers
    unique_authorid = np.unique(np.array([elm for singleList in \
            df[author_list_colname].values.tolist() for elm in singleList]))

    #Construct network based on unique_authorid and create edges of coauthors
    g = nx.Graph() # create network object
    g.add_nodes_from(unique_authorid) # add nodes of unique author IDs

    # add edges to the graph
    # coauthors have the edges
    print("Generating edges...")
    for idx, paper in tqdm(df.iterrows(), \
            total=df.shape[0]):
        author_list = paper[author_list_colname]
        # generate all the combinations of coauthors
        author_comb = list(combinations(author_list, 2)) 
        # iterate over all the author combinations (links)
        for link in author_comb:
            # weight: the number of collaborations
            if g.has_edge(*link):
                g.edges[link[0], link[1]]['weight'] += 1
            else:
                g.add_edge(link[0], link[1])
                g.edges[link[0], link[1]]['weight'] = 1

    print("Network summary: {}".format(nx.info(g)))

    return g


if __name__ == "__main__":
    print("Loading datasets...")
    covid19_paper_filename = "../dataset/COVID19_papers_modeling_final.csv"
    authorid_pub_database_filename = "../dataset/COVID19_authorid_pub_database.csv"

    covid19_paper_df = pd.read_csv(covid19_paper_filename)
    authorid_pub_df = pd.read_csv(authorid_pub_database_filename)

    ### COVID-19 collaboration network
    authorid_lst_eval = covid19_paper_df['author_id_modified'].apply(lambda x:eval(x))
    covid19_paper_df = covid19_paper_df.assign(author_id_modified = authorid_lst_eval)

    print("Constructing COVID-19 collaboration network...")
    covid19_g = construct_covid19_network(covid19_paper_df, "author_id_modified")

    # write COVID-19 collaboration network as edgelist
    nx.write_weighted_edgelist(covid19_g, "../dataset/covid19_collab_network.edgelist")
    # write the graph object as pickle
    outfile = open("temp/covid19_collab_g.pickle", "wb")
    pickle.dump(covid19_g, outfile)
    outfile.close()

    ### previous publication collaboration network
    # convert the author ID types into string
    authors_eval = authorid_pub_df['authors'].apply(lambda x: eval(x))
    authorid_pub_df = authorid_pub_df.assign(authors = authors_eval)

    print("Constructing previous publication network...")
    # select the Scopus ID and authors columns
    pub_scopusid_authors = authorid_pub_df[["scopus_id", "authors"]]
    # get the unique scopus IDs of previous publications and their authors
    unique_scopusid_authors = pub_scopusid_authors.drop_duplicates(subset="scopus_id")
    unique_scopusid_authors.reset_index(inplace=True, drop=True)

    prev_collab_g = construct_covid19_network(unique_scopusid_authors, "authors")

    # write COVID-19 collaboration network as edgelist
    nx.write_weighted_edgelist(prev_collab_g, "../dataset/prev_collab_network.edgelist")
    # write the graph object as pickle
    outfile = open("temp/prev_collab_g.pickle", "wb")
    pickle.dump(prev_collab_g, outfile)
    outfile.close()







