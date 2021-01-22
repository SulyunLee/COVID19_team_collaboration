
import pandas as pd
import numpy as np

if __name__ == "__main__":

    # authorid_affil_filename = "../dataset/COVID19_authorid_info_database.csv"
    authorid_affil_filename = "../dataset/FLU_authorid_info_database.csv"
    hofstede_country_score_filename = "../dataset/hofstede_country_scores.csv"

    authorid_affil_df = pd.read_csv(authorid_affil_filename)
    country_score_df = pd.read_csv(hofstede_country_score_filename)

    affiliation_country = authorid_affil_df.affiliation_country.dropna()
    unique_affil_country = affiliation_country.unique()

    print("Number of unique affiliation countries: {}".format(unique_affil_country.shape[0]))
    
    for country in unique_affil_country:
        if not country_score_df['country'].str.contains(country).any():
            print(country)

    # unique_affil_country_df = pd.DataFrame(unique_affil_country, columns=['country'])
    # unique_affil_country_df.to_csv("../dataset/hofstede_country_scores.csv", \
            # index=False, encoding='utf-8-sig')

    # left join to combine country scores
    
    
    
    
    
    
    
    
    

