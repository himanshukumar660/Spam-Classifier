import os
import numpy as np
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.metrics import accuracy_score

#Generate a Pandas Training DataFrame
TrainingData = pd.DataFrame()
#Generate a Pandas Testing DataFrame
TestingData = pd.DataFrame()

#Populate the Data Frame using the Training Data Model 

def populateModel(dataframe,path):
    for root,dirs,filenames in os.walk(path):
        for filename in filenames:
            filePath = os.path.join(root, filename)
            dataframe = dataframe.append(pd.read_csv(filePath))
    return dataframe

TrainingData = populateModel(TrainingData,'/Users/himanshukumar/Documents/Projects/Machine Learning/Spam Classifier/YouTube-Spam-Collection-v1/Train')

mapping = {0:"Ham",1:"Spam"}
TrainingData['CLASS'] = TrainingData['CLASS'].map(mapping)


#CountVectorizer will convert a collection of text Documents into a matrix of token Counts
vectorizer = CountVectorizer()
Counts = vectorizer.fit_transform(TrainingData['CONTENT'].values)
targets = TrainingData['CLASS'].values

classifier = MultinomialNB()
res = classifier.fit(Counts,targets)

#Now Call PopulateModel for Test Data
TestingData = populateModel(TestingData, '/Users/himanshukumar/Documents/Projects/Machine Learning/Spam Classifier/YouTube-Spam-Collection-v1/Test')
TestingData['CLASS'] = TestingData['CLASS'].map(mapping)

#Calculate the count tokens for each comments using transform
testingCounts = vectorizer.transform(TestingData['CONTENT'].values)

#Calculate the predictions of the testing Data
predictions = classifier.predict(testingCounts)

#Store the Original results in  a variable called original 
original = TestingData['CLASS'].values

#Print The Accuracy of the Results produced
Accuracy = accuracy_score(predictions, original)
print ("Accuracy : {}".format(Accuracy))