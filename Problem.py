# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 19:36:22 2018

@author: amr
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 31 04:16:45 2018

@author: amr
"""

#using json 
#import json 
#
#with open('train_articles.json') as j:
#     data = json.load(j)

import pandas as pd #to read json file to pandas dataframe 
dataset = pd.read_json('file:///D:/summer%202/q1/train_articles.json',orient='columns') #use your url  

dataset = dataset.sort_index()
dataset = dataset.reset_index(drop=True)

# Cleaning the texts
import re  #liberary for remove all things that i don't need like punction numbers 
import nltk
nltk.download('stopwords')  #download stopwords file
from nltk.corpus import stopwords #grasping stop words file to use 
from nltk.stem.porter import PorterStemmer  #liberary for stemming 

#clean the data 
corpus = []
for i in range(0, 3426):
    title = re.sub('[^a-zA-Z]', ' ', dataset['title'][i]) #remove any thing that not a - z or A-Z and replace it with space
    title = title.lower() #make every letter small
    title = title.split() #split the string to list
    ps = PorterStemmer()
    title = [ps.stem(word) for word in title if not word in set(stopwords.words('english'))] #stem every word and if it's not in stopwords put it again in the list, we use set because it has faster algoirthms for huge data 
    title = ' '.join(title) #back again to string 
    
    body =  re.sub('[^a-zA-Z]', ' ', dataset['body'][i])
    body = body.lower()
    body = body.split()
    ps = PorterStemmer()
    body = [ps.stem(word) for word in body if not word in set(stopwords.words('english'))]
    body = ' '.join(body)
    
    corpus.append(title+' '+body) #concatenate the title and the body as a one string 

#bag words 
from sklearn.feature_extraction.text import CountVectorizer #to make sparse matrix then tokenization 
cv = CountVectorizer(max_features = 22700)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values    

for k in range(3426):
   for i in range(len(y[k])):
      y[k][i] = y[k][i].replace(" ","")
   y[k] = ' '.join(y[k])   
   
cv = CountVectorizer()
y = cv.fit_transform(y).toarray()    
#########

#######################
#to get index of tags again 
ytags = dataset.iloc[:, 1].values
#for k in range(3426):
#   for i in range(len(ytags[k])):
#      ytags[k][i] = ytags[k][i].replace(" ","")
#   ytags[k] = ' '.join(ytags[k])  
   
cvtags = CountVectorizer()
temptagsparse = cvtags.fit_transform(ytags).toarray()    

########################

#train the data 
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X, y)

###################################################################################################

#get unlabeled atricles for testing
unlabeleddataset = pd.read_json('file:///D:/summer%202/q1/test_articles.json',orient='columns') #use your url
unlabeleddataset = unlabeleddataset.sort_index()
unlabeleddataset = unlabeleddataset.reset_index(drop=True)

unlabelcorpus = []
for i in range(0, 1143): 
    title = re.sub('[^a-zA-Z]', ' ', unlabeleddataset['title'][i])
    title = title.lower()
    title = title.split()
    ps = PorterStemmer()
    title = [ps.stem(word) for word in title if not word in set(stopwords.words('english'))]
    title = ' '.join(title)
    
    body =  re.sub('[^a-zA-Z]', ' ', unlabeleddataset['body'][i])
    body = body.lower()
    body = body.split()
    ps = PorterStemmer()
    body = [ps.stem(word) for word in body if not word in set(stopwords.words('english'))]
    body = ' '.join(body)
    
    unlabelcorpus.append(title+' '+body)

cv = CountVectorizer()
X_test = cv.fit_transform(unlabelcorpus).toarray()

y_pred = classifier.predict(X_test) #predict tag for the unlabeled atricles 

unlabeleddataset['tags'] = " "
tags = []
for i in range(1143):
     tags.append(cvtags.inverse_transform(y_pred[i]))   
    
print(cvtags.inverse_transform(y_pred[0]))

#tags for each atrticle 
for i in range(1143):
      print (tags[i])

    



  


     
     



