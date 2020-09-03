# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

# In-built imports
import pickle
import os
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from azureml.core.run import Run

from utils import mylib

# New imports
import pandas as pd 
import numpy as np 
import string 
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.feature_extraction.text import CountVectorizer
import nltk
#nltk.download()
from nltk.corpus import stopwords
stop = stopwords.words("english")
from nltk.stem.porter import *
from nltk.stem.porter import PorterStemmer

# Importing csv file and pre-processing
initial = pd.read_csv("initial_data.csv")
initial.pattern = initial.pattern.fillna("none")
initial.is_salary = initial.is_salary.fillna(0)

# remove whitespace and special characters/numbers
initial['pattern'] = initial['pattern'].str.replace("\s{2,}", " ")
initial['pattern'] = initial['pattern'].str.replace("[^a-zA-Z]", " ")

# remove stopwords
initial['pattern'] = initial['pattern'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop)) 

# stem patterns
stemmer = PorterStemmer()
tokenized_pattern = initial['pattern'].apply(lambda x: x.split())
tokenized_pattern = tokenized_pattern.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
for i in range(len(tokenized_pattern)):
    tokenized_pattern[i] = ' '.join(tokenized_pattern[i])
initial['pattern'] = tokenized_pattern

def remove_duplicates(pattern):
    pattern = pattern.split()
    ulist = []
    [ulist.append(x) for x in pattern if x not in ulist]
    unique = ' '.join(ulist)
    return unique
  
## remove repeated words 
pattern_unique = []
for pattern in list(initial.pattern): 
    pattern_unique.append(remove_duplicates(pattern))
initial.pattern = pattern_unique
  
# remove date and number strings 
initial['pattern'] = initial['pattern'].str.replace("yymmdd| yymmdd|yymmdd ", "")
initial['pattern'] = initial['pattern'].str.replace("nnn| nnn|nnn ", "")

initial.to_csv("temp.csv")
temp = pd.read_csv("temp.csv")
temp.pattern = temp.pattern.fillna("none")

#final_data = temp.iloc[:,4:]
df = temp.iloc[:,4:]

# Rest of code - integrating auto/default code with my own training:
os.makedirs('./outputs', exist_ok=True)

#X, y = load_diabetes(return_X_y=True)
X, y = df['description'], df['is_salary']

run = Run.get_context()

#Training and Testing Sets
#X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Converting Text to Numbers
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X_train = tfidfconverter.fit_transform(X_train).toarray()

X_test = tfidfconverter.transform(X_test).toarray()

#Their default code now
data = {"train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test}}

# list of numbers from 0.0 to 1.0 with a 0.05 interval
alphas = mylib.get_alphas()

#Training Text Classification Model and Predicting Salary Items
from sklearn.ensemble import RandomForestClassifier

#for alpha in alphas:
    # Use Ridge algorithm to create a regression model
    #reg = Ridge(alpha=alpha)
    #reg.fit(data["train"]["X"], data["train"]["y"])
    
# Using RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test)
#y_pred = classifier.predict(data["test"]["X"])

    #preds = reg.predict(data["test"]["X"])
    #mse = mean_squared_error(preds, data["test"]["y"])
    run.log('alpha', alpha)
    run.log('mse', mse)

# Save model in the outputs folder so it automatically get uploaded when running on AML Compute
    #model_file_name = 'ridge_{0:.2f}.pkl'.format(alpha)
    #with open(os.path.join('./outputs/', model_file_name), 'wb') as file:
    #    pickle.dump(reg, file)

import pickle
model_file_name = 'finalized_model.sav'
with open(os.path.join('./outputs/', model_file_name), 'wb') as file:
    pickle.dump(reg, file)


    #print('alpha is {0:.2f}, and mse is {1:0.2f}'.format(alpha, mse))
