
# Movie-Schedule-Optimization-Using-Sentimental-Analysis-of-Movie-Reviews

## Introduction
The triumph of a movie’s theatrical run is greatly 
depends on the selection of theaters and showtimes. But 
current methods frequently overlook audience reactions 
in real time, which results in an ineffective use of 
resources. Many popular movies have trouble finding 
theaters to screen them, and too many movies with bad 
reviews end up in theaters, frustrating moviegoers and 
hurting distributors' bottom lines. 
Our study uses sentiment analysis on first-day movie 
reviews to examine a fresh solution to this problem. By 
compiling and analyzing these reviews, we can predict 
how well or poorly a film will do at theaters. This data
driven approach may allow us to recommend 
adjustments to the number of theaters and showtimes for 
each movie.In order to categorize the tone of 
evaluations, we utilize several machine learningTo 
classify the sentiment of reviews, we employ a number 
of machine learning methods, including SVM, Logistic 
Regression, Naïve Bayes and KNN. 
The model Multinomial Naive Bayes is used for final 
recommendations because of its greater accuracy. Our 
goal in incorporating sentiment analysis into the 
decision-making process is to make movie scheduling 
more effective. This would not only help distributors cut 
losses but also improve audience demand. This 
concept's dynamic and adaptable venue allocation 
mechanism benefits both moviegoers and the film 
distribution business.

## Problem Statement and Project Objective
Firstly, we would like to understand whether ranking on movie rating websites will affect potential audiences’ willingness to watch the movie. If a movie has a high rank, it will attract more audiences to go to the theaters to watch this movie. However, for newly released movies, the website should wait for days to conclude a rating, and cinemas
apparently could not wait. They need to conclude how the first batch of audiences react on SNS and use this information to arrange their screenings. Thus, the next step is to predict the rating. We decided to formulate a model to analyze audiences’sentiment by their text comments on the Internet. In this way, cinemas can easily gain a quick conclusion from people's comments and make decisions.

## Methodology
We firstly apply linear regression to gain the effects of Internet rating on movie revenue. Then we use TextBlob to explore the polarity and subjectivity of movie review comments. After that we evaluate the plain accuracy and confusion matrix of four models: Logistic Regression, SVM, Naive Bayes and K-nearest Neighbors Classifier, and choose Naive Bayes as our final model.

### Relationship between Internet Rating and Movie Revenue
To confirm our hypothesis that Internet comments are affecting the revenue of movies, we did a simple linear regression.

#### OLS Regression Result
The coefficient of vote_average is 0.4747 and p value is 0, which means vote_average has a significant
positive correlation with movies’ revenue. The result matches our hypothesis and we can move to the next
step.

### Sentimental Analysis using TextBlob
We wanted to know whether the movie comments are consistent with the review scores, so we calculated the polarity scores to see if comments with positive scores are given higher review scores. Moreover, we calculated the subjectivity score to see if the subjectivity score is higher when the absolute value of the polarity score is high.

#### Sentimental Analysis Criteria
● Polarity ~ [-1,1], 0 is neutral, +1 is positive, -1 is negative.
● Subjectivity ~ [0,1], 0 is objective and 1 is subjective.

#### Feature engineering
<img width="476" alt="feature-engineering" src="https://user-images.githubusercontent.com/88580416/145959178-a20d9c09-6cbb-4cb2-a5a8-ad2a5dfb403d.PNG">

#### Data Encoding
Encode review_rank to numeric values
Encode review_type to dummy variables
<img width="469" alt="rotten_tomato-df" src="https://user-images.githubusercontent.com/88580416/145959326-54e9a999-5e7c-4e6b-b565-b198957981ee.PNG">

#### EDA and Preliminary Results
This scatter plot illustrates the polarity score distribution within each review score. As you can see, there are more data points that have a positive polarity score in higher review scores and there are less data points that have a positive polarity score in lower review scores. For example, At review score 12, which corresponds to review score A in the original dataset, we see there are comparatively less data points that have a negative polarity score.

<img width="278" alt="polarity-scatter" src="https://user-images.githubusercontent.com/88580416/145959538-4e436b4c-32a1-4852-9ebd-289305ca5622.PNG">

Furthermore, we compute the average polarity score in each review score range. Interestingly, the higher the review score, the higher the polarity score. We concluded that the polarity score, which is calculated from the review comments, and review score, which is ranked initially by commenters, have a positive correlation.

<img width="266" alt="average-lineplot" src="https://user-images.githubusercontent.com/88580416/145959690-8e54fdb5-e4b7-4b7f-9f84-855422cbdfdc.PNG">

We also wanted to know if the polarity score calculated from movie comments has a significant effect on review score, so we ran an OLS regression model. The polarity score has a positive coefficient and a 0.0000 p-value, meaning that the polarity score has a significant effect on the review score and they are positively correlated.

<img width="280" alt="regression" src="https://user-images.githubusercontent.com/88580416/145959816-2fde8dd3-af08-4e04-b8b7-20387afc8e02.PNG">

We also plot all the data points using their polarity score and subjectivity score calculated from the review comments. We concluded that more polar comments tend to be more subjective.

<img width="231" alt="scatter-01" src="https://user-images.githubusercontent.com/88580416/145960033-14b9544f-b2e6-4bce-a479-cbde73e96cda.PNG">

In this graph, we filtered out the comments that are too subjective (subjectivity score > 0.8) and the movies that have only one comment. Therefore, we have all the data points that have a subjectivity score <= 0.8 and have 2 or more review comments for each movie. Then, we calculated the polarity score of the review comments for each movie and plot against the review score. We saw a slightly positive correlation between the average polarity score for each movie and review scores.

<img width="248" alt="scatter-02" src="https://user-images.githubusercontent.com/88580416/145960093-18725a06-e0ae-4e0e-808c-9f191e2f3b60.PNG">

### Predict movie sentiments with classification models
Using the same dataset as sentimental analysis (Rotten tomatoes review), we ran several classification models to see if we can use movie comments to predict the sentiments of viewers. Sentiments are calculated from review type. If the movie has a review score greater than or equal to 9 (B-), then the movie has a sentiment of 1, else 0. We input “movie comments” as our independent variables and used “sentiment” as our dependent variable.

#### Data Preprocessing method
How can we input review contents as independent variables? We first tokenized the sentences into single words. Second, we took out symbols and words without specific positive or negative meaning, such as “with”, “should”, or “these” from the review content. Then, we applied two algorithms: count vectorizer and tf-idf vectorizer, to convert review_content into vectors (meaningful representation of numbers) that can be put into our classification models.

#### The count vectorizer & the tf-idf vectorizer
The count vectorizer considers the frequencies of words in a sentence, while tf-idf vectorizer considers both the frequencies a word appears in a sentence and the number of sentences the word appears in.

#### Classification models’ results
We ran 4 classification models, including Logistic Regression, SVM, Multinomial Naive Bayes, and K-nearest neighbors classifier. The accuracy rates calculated from the count vectorizer (BOW) generally have a better performance. Overall, the Multinomial Naive Bayes model has a comparatively higher accuracy rate, so we chose it as our final model.

<img width="468" alt="classification-models" src="https://user-images.githubusercontent.com/88580416/145960843-81ad88ef-09d6-436b-8d09-7f2e54e41636.PNG">

#### Confusion Matrix
In the test dataset, there are 17,498 data points, with 11,300 true positive sentiments and 6,198 true negative sentiments. The data is unbalanced, with more positive sentiments (about 64.58% are positive sentiments). Therefore, we further looked at the confusion matrix on the models that are initially input with count-vectorizer vectors. In SVM, K-nearest neighbors classifier, and Logistic Regression models, almost all the sentiments are predicted as positive (sentiment = 1), and only the Multinomial Naïve Bayes model has a slightly more accurate prediction with 16356 positive and 1142 negative sentiment predictions.

<img width="346" alt="confusion-matrix" src="https://user-images.githubusercontent.com/88580416/145960994-84f63dae-39c5-475e-9bac-ccbfc9a9d84b.PNG">

#### Word Cloud
Moreover, we tried to understand the most frequent positive or negative review words used by reviewers. We used the “WordCloud” package and the results are shown in the below figure.

<img width="452" alt="word-cloud" src="https://user-images.githubusercontent.com/88580416/145961175-2a590faf-3bf6-44da-9693-11dffad6e3cf.PNG">

## Business Application
In a real life setting, film investors and stakeholders often want to make the cinematic arrangements such as ticket prices, ticket availability, and most importantly, number of showtimes and locations of different cinemas the movie will be primarily aired in. It is crucial for them to understand the market demand for the film they invested in in order to maximize profit, and a common way to do that is through premieres and test screenings. We believe that stakeholders and investors could utilize the viewers’ comments from
the premieres and test runs, and perform sentiment analysis on them to extract the initial responses. In order to arrive at better decisions, we could combine viewers’ demographics with their comments’ sentiment analysis, and figure out which groups respond the best to the film, and which ones detest it or find it somewhat offensive. This way investors of the film could choose to make smarter decisions such as not airing it too much, air it in specific states where the demographics are mostly likely warm to the
content, air it during times where the demographics would visit the cinemas the most, etc. The editors of the film could even conduct some further editing and blurring, to make sure that the movies are welcome to the wanted target audience, and therefore generate the most revenue out of each film.

## Conclusion and Further Navigations
In conclusion, we have determined that the Multinomial Naive Bayes model has the most accurate predictions in comparison with the other models in the setting of predicting sentiments with movie comments. This could be used as a preliminary analytical tool for stakeholders to predict the gross box office of the film, and therefore better arrange the screening times and locations. However, there exists a few areas that could be navigated further. To start off, we ruled out comments that we considered as “too subjective” and “too polarized”, but there are films that have such a strong and unique characteristic that only appeal to a certain crowd. Some comments might love it and some might detest it, but both need to be included to arrive at an accurate and relevant prediction. This accuracy issue might also occur as we tokenized the words. We could easily break down a sentence into words that didn’t convey its original implication. For example, a “huge mistake if you don’t watch it over 100 times” comment would give the wrong idea. This could be further improved if we can incorporate more phrases or short sentences in the model to start with, and therefore interpreting a wider range of sentiments. Another important issue to consider is that, during the initial analysis of our data, we recognized that sentiments are in values of only 0 and 1, that is, identifying emotions that are purely positive and purely negative. We could consider that words and emotions have different levels of positivity and negativity, and only considering a comment section’s count of words and counts of sentences in which they appear could throw us off quite a bit. To further improve the accuracy rate of our classification models, we can add a more refined score to reflect comment’s emotions, and eventually produce a movie classification that is more than just “good” and “bad”.

## Reference
[1] Baid, P., Gupta, A., & Chaplot, N. (2017). Sentiment 
Analysis of Movie Reviews using Machine Learning 
Techniques. International Journal of Computer 
Applications, 
179(7). https://doi.org/10.5120/ijca2017916005  

[2] Kalaivani, P., & Shunmuganathan, K. L. (2013). 
Sentiment Classification of Movie Reviews by 
Supervised Machine Learning Approaches. Indian 
Journal of Computer Science and Engineering, 4(4), 
285-290. 

[3] Purnomoputra, R. B., Adiwijaya, & Wisesty, U. N. 
(2019). Sentiment Analysis of Movie Reviews using 
Naïve Bayes Method with Gini Index Feature Selection. 
Journal of Data Science and Its Applications, 2(2), 85
94. https://doi.org/10.34818/JDSA.2019.2.36  

[4] Sulthana, R. A., Jaithunbi, A. K., Harikrishnan, H., 
& Varadarajan, V. (2022). Sentiment analysis on movie 
reviews dataset using support vector machines and 
ensemble learning. International Journal of Information 
Technology and Web Engineering, 
https://doi.org/10.4018/IJITWE.311428.  
17(1). 

[5] Kulkarni, E. G., & Pujar, A. M. (2021). Intelligent 
framework for sentiment analysis of movie reviews - A 
comprehensive survey. Journal of Emerging 
Technologies and Innovative Research, 8(6), 738-742. 
http://www.jetir.org/papers/JETIR2106802.pdf  

[6] Bodapati, J. D., Veeranjaneyulu, N., & Shaik, S. 
(2019). Sentiment analysis from movie reviews using 
LSTMs. Ingenierie des Systemes d'Information, 24(1), 
125-129. https://doi.org/10.18280/isi.240119  

[7] Lee, & Pang, B. (2008). sentiment analysis as well 
as opinion mining. Principles and Developments in 
Information Retrieval, 2(1-2), 1–135. 
(doi.org/10.1561/1500000011)  

[8] Wiley Interdisciplinary Reviews: Data Mining and 
Knowledge Discovery, 8(4), e1253. Zhang, L., Wang, 
S., & Liu, B. (2018). Deep learning for sentiment 
analysis: A survey. https://doi.org/10.1002/widm.1253  

[9] Asur, S. and Huberman, B. A. (2010). using social 
media to make future predictions. The 2010 
IEEE/WIC/ACM International Conference on Web 
Intelligence and Intelligent Agent Technology, 
Proceedings, 492-499. 
doi.org/10.1109/WI-IAT.2010.63  

[10] In 2003, Hatzivassiloglou, V., and Yu, H. 
Identifying the polarity of opinion sentences and 
separating opinions from facts are two steps towards 
providing an opinion-based answer. Empirical Methods 
in Natural Language Processing (EMNLP) Conference 
Proceedings, 2003, 129–136. [W03-1017] available at 
https://www.aclweb.org/anthology  

[11] In 2010, Tumasjan, A., Welpe, I. M., Sandner, P. G., 
and Sprenger, T. O. Predicting elections with Twitter: 
What 140 characters convey about political mood. The 
Fourth International AAAI Conference on Social Media 
and Weblogs, Proceedings, 178–185. 
This 
article 
can 
be 
viewed at
https://ojs.aaai.org/index.php/ICWSM/11001 

Video link: https://youtu.be/8UCfaQtp9lY
