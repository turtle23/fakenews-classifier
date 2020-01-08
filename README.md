# fakenews-classifier
A Multi-Class Classification Model using Keras Functional API  
(Project has been created for learning purpose only)

For information about Keras Functional API visit https://keras.io/getting-started/functional-api-guide/

Dataset and information about the fake news challenge can be found on http://www.fakenewschallenge.org/

For word embedding layer (Fast-text):

Download pre-trained word vectors 'wiki-news-300d-1M.vec.zip' available on 
https://fasttext.cc/docs/en/english-vectors.html



A Headline and a Body text either from the same news article or from two different articles are given.

Classify the stance of the Body text relative to the claim made in the headline into one of the four categories:

  Agree,  Disagree,  Discuss,  Unrelated
  
  
Holistic view of dataset

|agree	|disagree	|discuss	|unrelated	| Total | 
|-----  |-------- |  -------|  ---------| ------|
|3678	  |840	    |8909     |	36545	    | 49972 |
|7%	    |1.6%	    |17%	    |73%	      |100%   |

confusion matrix on test data

|   -   |Predicted agree |Predicted disagree |Predicted discuss |Predicted unrelated|
|------ |------|----------| --------|----------|
|Actual agree|    4 |  1       | 1       |    13    |
|Actual disagree|1   |3         |  0      |    5      |
|Actual discuss|2   |0         |7        |3          |
|Actual unrelated|1  |0         |1        |180        |

Prediction accuracy = 0.7760 and Misclass = 0.2240




 
