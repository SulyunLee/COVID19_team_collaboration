# Team Collaboration for COVID-19 Research

## Summary
This is a research project that explores the researchers' team collaboration patterns when a new research field emerges, such as COVID-19.
We measured the research team's knowledge continuity that indicates how much researchers' current publications of COVID-19 are similar to their prior expertise. Then, we used the research team's knowledge continuity and other control variables to predict the team performance using three indicators: impact, breakthrough, and novelty. We used explanatory models, such as logistic regression and linear regression, that give the coefficients for the variables in explaining the association with the team performance. We compared the models constructed on the COVID-19 publication samples to the models constructed on the established research field of influenza to examine how team knowledge continuity patterns differ between the emerging and established fields. Through our experiments, we discovered the unique mechanisms of collaborative patterns among research teams in the new field of COVID-19 and provided meaningful suggestions in effectively combating COVID-19.


## Dataset
We used the COVID-19 Open Research Dataset (CORD-19) that provides up-to-date publications about coronavirus-related diseases, such as COVID-19 and SARS-CoV-2 (https://allenai.org/data/cord-19). In addition to this database, we used Scopus, PubMed, and PMC APIs to collect additional information, such as historical publication records of authors.

## Model description
### Dependent Variable
Three innovation indicators: 
* *impact* : the average number of the forward citation counts of a publication per month
* *breakthrough* : a binary indicator of whether a publication was highly cited among other publications published in the same month
* *novelty* : how much a publication has novel content

### Independent Variable
* *Team knowledge continuity*: <br>
The similarity between the researcher's prior expertise and the COVID-19 focal paper.
We used the topic modeling technique, Latent Dirichlet Allocation (LDA) (Blei et al. 2003), to compute the latent topics of each paper published by authors who published at least one COVID-19 publication. Then, we calculated the similarity between the focal paper's topic distribution and the author's prior topic distribution computed by their previous publications.

### Control Variables
|Variable|Explanation|
|--------|-----------|
|Expertise Similarity|The average similarity among team members' prior expertise|
|New collaboration rate|The percentage of new co-authorship links|
|Disparity of h-index|Gini-coefficient of team members' h-indices|
|Similarity of cultural background|Similarity of culture representations among team members' countries of affiliations|
|Variance of knowledge continuity|Variance of team knowledge continuity among team members|
|Maximum h-index|Maximum h-index of team members|
|Team size|The number of team members|
|Practical affiliation rate|The percentage of authors who are practitioners|
|Publication month|Month of publication date|
|Topic distribution|LDA topic distributions of focal papers|
