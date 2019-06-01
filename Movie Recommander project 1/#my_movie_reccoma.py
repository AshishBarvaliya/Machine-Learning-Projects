import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
    
r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3))

m_cols = ['movie_id', 'title']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

ratings =pd.merge(movies,ratings)

#pivot table
movieRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')

#extract users who rated star war
starwarrate = movieRatings['Star Wars (1977)']

similarMovies = movieRatings.corrwith(starwarrate)
similarMovies = similarMovies.dropna()
df = pd.DataFrame(similarMovies)
df.head(10)
movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
movieStats.head()
popularMovies = movieStats['rating']['size'] >= 100
movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)

df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['0']))

sortdf=df.sort_values(['0'], ascending=False)
