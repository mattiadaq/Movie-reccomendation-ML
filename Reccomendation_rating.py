# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 16:43:39 2025

@author: matti
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# the data used had some NaN - cleans the data
df = pd.read_csv("movie_dataset.csv")
features = ['keywords', 'cast', 'genres', 'director']
for feature in features:
    df[feature] = df[feature].fillna('')


# Clean vote_average to make sure it's numeric
df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')  # convert invalid strings to NaN
df['vote_average'] = df['vote_average'].fillna(0)  # replace NaNs with 0 as they dont have a rating

# use previous step predictors
def combined_text(row):
    return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']
df['combined_text'] = df.apply(combined_text, axis=1)

# use this for similarity measurements
cv = CountVectorizer()
text_matrix = cv.fit_transform(df['combined_text'])

# Normalize vote_average to 0–1 to get comparisons
scaler = MinMaxScaler()
df['vote_scaled'] = scaler.fit_transform(df[['vote_average']])

# Step 5: Combine similarity
text_similarity = cosine_similarity(text_matrix)

# Add numeric similarity manually
# Create a (n x n) matrix for vote_average similarity using subtraction
vote_diff = np.abs(df['vote_scaled'].values.reshape(-1, 1) - df['vote_scaled'].values.reshape(1, -1))
vote_similarity = 1 - vote_diff  # Higher when values are closer

# choose the weighting between factors and score
combined_similarity = 0.2 * text_similarity + 0.8 * vote_similarity

# Step 6–8: Search
movie_user_likes = "2001: A Space Odyssey"
movie_index = df[df.title == movie_user_likes].index[0]

similar_movies = list(enumerate(combined_similarity[movie_index]))
sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

print(f"\nTop 15 movies similar to '{movie_user_likes}':\n")
for i, (index, score) in enumerate(sorted_similar_movies[:16]):
    print(df.iloc[index]['title'])
