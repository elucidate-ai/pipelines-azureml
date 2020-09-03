import json
import numpy as np
import pickle
from sklearn.linear_model import Ridge
from azureml.core.model import Model
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from utils import mylib

# Other imports just in case
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Code to allow for input_sample and output_sample

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

#Importing original csv file that will not be modified
initial_unmodified = pd.read_csv("initial_data.csv")

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

# Back to default code
def init():
    global model
    model_path = Model.get_model_path('diabetes-model')

    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # For demonstration purposes only
    #print(mylib.get_alphas())   (#'d this)

#input_sample = np.array([[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
#output_sample = np.array([3726.995])

input_sample = initial_unmodified[0]
#input_sample = X[0]
output_sample = y[0]

@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        # you can return any datatype as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
