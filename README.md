# Text-Mining---Cluster-Analysis

In order to determine whether Shakespeare was a single genius or team of playwrights, we implemented several tools learned in class to inspect the master pieces. 

To begin with, we decided to check whether there was any particular sentiment pattern both across each text and across all of the pieces. We applied Textblob to each of the lines and calculated the polarity score proportion of each sentiment (Positive, Negative, Neutral) to each piece. We can see the results pasted in a bar plot; the “Neutral” sentiment dominates the results and none of the texts exhibit a clear “Positive” or “Negative” tendency. 

Afterwards, we performed a similar analysis using VADER sentiment analysis tool across each text. Once again, we see no clear pattern or tendency (please see the Jupyter code for the rest of the texts).

<img width="793" alt="Screen Shot 2021-04-20 at 7 26 28 PM" src="https://user-images.githubusercontent.com/34326154/115471728-d1eacd00-a20e-11eb-9236-9b9706a9a5cb.png">

<img width="789" alt="Screen Shot 2021-04-20 at 7 28 39 PM" src="https://user-images.githubusercontent.com/34326154/115471862-0f4f5a80-a20f-11eb-9e31-67ce7a31d023.png">

Since the first step of the analysis did not allow us to reach any conclusion, we decided to apply TFIDF (Term Frequency Inverse Document Frequency). This statistical technique will allow us to translate Shakespeare pieces into matrices by and labeling the most relevant words into numbers and creating a big matrix. This should allow us to detect the most distinctive words in each text and then run the cosine similarity analysis. The cosine analysis between two pieces will output a number closer to one when they are similar. As can see from the cosine_similarity matrix, the results show no clear no distinctive patters among texts. 

We continued our analysis by running PCA on the matrix that resulted from running TFIDF. PCA is a linear dimensionality reduction method that will allow us to build orthogonal projections that most of the variability in the matrix under study. By plotting the projections on the scatter plots, we were able to detect potential outliers. PCA analysis shows suggests that the following plays are outliers:

•	A Comedy of Errors

•	Julius Caesar

•	Antony and Cleopatra

•	Titus Andronicus


<img width="758" alt="Screen Shot 2021-04-20 at 7 33 34 PM" src="https://user-images.githubusercontent.com/34326154/115472072-70772e00-a20f-11eb-831e-4bd1ef13035a.png">


<img width="764" alt="Screen Shot 2021-04-20 at 7 33 44 PM" src="https://user-images.githubusercontent.com/34326154/115472151-a87e7100-a20f-11eb-8353-fc1140b38366.png">


Therefore, we run K-Means to confront and confirm this evidence. K-means is an unsupervised learning method. It is used to look for patterns in data when there is no particular target feature, or dependent variable. K-means clustering is a simple and elegant approach for partitioning a data set into K distinct, non-overlapping clusters. K-means problem is solved using Lloyd’s algorithm, which partitions the observations into K clusters such that the total within-cluster variation, summed over all K clusters, is as small as possible.  Silhouette scores can be used to help evaluate the appropriate number of clusters that are truly in the data.

After running K-means on our dataset we found that number of clusters that reach the highest silhouette score (0.53) is three. 

We then applied the Hierarchical analysis. The result of hierarchical clustering is a tree-based representation of the objects, which is also known as dendrogram. Each node represents a group.

<img width="1078" alt="Screen Shot 2021-04-20 at 7 37 29 PM" src="https://user-images.githubusercontent.com/34326154/115472285-f4311a80-a20f-11eb-8dd3-ee2fb26fcf15.png">


Both clustering techniques show similar results. We decided to explore further into these three clusters and see if there is any writing pattern relevant in each of them.

By looking at the dendrogram, we decided to re-run K-means Hierarchical analysis again but this time excluding the play called “A Comedy of Errors”. Since K-Means reported the highest silhouette score with three clusters, we cut the H-tree by three. This second process allowed us to extract four clusters, which we will discuss further below.

Before doing that, we wanted to check whether there was any hint that these clusters were associated in time. To do this, we plotted the PCA projections and labeled each play using the year in which they were written according to Wikipedia. As we can see from the plots below, there does not seem to be any relation between the year in which the plays were published and the clusters we obtained above. However, it is clear that if it is true that Shakespeare was a single person, it is clear that he had a great talent since the majority of the plays were written in a short span of time.

<img width="511" alt="Screen Shot 2021-04-20 at 7 39 21 PM" src="https://user-images.githubusercontent.com/34326154/115472387-23e02280-a210-11eb-9eab-390922c00aec.png">


Finally, we realized that the resulting clusters shared common traits. Each cluster has a broad theme associated. They are either tragedies, stories related to kings, love stories or comedies. There is evidence to suggest that maybe Shakespeare was a group of people who focused on different writing styles. In order to get a deeper insight on this fact, I decided to explore run LDA in each of the cluster and check the dominant themes and words. The results can be seen below. 


Cluster 1: Tragedy
['Hamlet', 'Coriolanus', 'Cymbeline', 'Antony and Cleopatra', 'King Lear', 'Othello', 'Troilus and Cressida', 'A Winters Tale', 'Henry VIII', 'Alls well that ends well', 'Measure for measure', 'Loves Labours Lost', 'Merry Wives of Windsor', 'As you like it', 'Merchant of Venice', 'Julius Caesar', 'Much Ado about nothing', 'Twelfth Night', 'Pericles']


<img width="970" alt="Screen Shot 2021-04-20 at 7 40 25 PM" src="https://user-images.githubusercontent.com/34326154/115472688-bd0f3900-a210-11eb-851b-8223d34c0497.png">


Cluster 2: Kings

['Richard III', 'Henry V', 'Henry VI Part 2', 'Henry IV', 'Henry VI Part 3', 'Henry VI Part 1', 'Richard II', 'King John', 'Titus Andronicus', 'Timon of Athens', 'macbeth', 'The Tempest', 'A Midsummer nights dream']

<img width="961" alt="Screen Shot 2021-04-20 at 7 44 40 PM" src="https://user-images.githubusercontent.com/34326154/115472764-e4660600-a210-11eb-9758-17997efde92c.png">

 
Cluster 3: Love, arranged matrimonies,  

['Romeo and Juliet', 'Taming of the Shrew', 'Two Gentlemen of Verona']

<img width="965" alt="Screen Shot 2021-04-20 at 7 45 25 PM" src="https://user-images.githubusercontent.com/34326154/115472826-fe074d80-a210-11eb-9a0a-45e52f0486e0.png">

 Cluster 4: Comedy, short story

A Comedy of Errors

I could have continued to explore deeper and exclude words that do not seem to add information to the LDA analysis. Due to time constraint I have stopped the analysis here. We could explore in each text if there is any pattern in the different characters of the texts that belong to the same cluster. That could be done by first identifying the lines of each of this character and reorganize the LDA analysis by means of characters and not texts. The result of this analysis would show us whether characters share common patters. This could lead us to conclude whether writing patterns change among characters and clusters. 



