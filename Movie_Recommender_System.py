import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def index_title(index):
    return dataset[dataset.index==index]["title"].values[0]
    
def title_index(title):
    return dataset[dataset.title==title]["index"].values[0]
    
    
dataset = pd.read_csv('movie_dataset.csv')

features = ['director','genres','cast','keywords']

for feature in features:
    dataset[feature]=dataset[feature].fillna(' ')
    
def combine_features(row):
    try:
        return row['keywords']+row['genres']+row['cast']+row['director']
    except:
        print('Error'),row
        
dataset["Combined_Features"]=dataset.apply(combine_features,axis=1)

cv = CountVectorizer()

count_matrix = cv.fit_transform(dataset["Combined_Features"])

cosine_sim = cosine_similarity(count_matrix)
User_Liked_Movie = "Spectre"

Movie_index = title_index(User_Liked_Movie)

Similar_Movies = list(enumerate(cosine_sim[Movie_index]))

Sorted_Movies = sorted(Similar_Movies,key= lambda x:x[1] , reverse = True)

i=1
for movie in Sorted_Movies:
    print(index_title(movie[0]))
    i=i+1
    if i>11:
        break
    
    
