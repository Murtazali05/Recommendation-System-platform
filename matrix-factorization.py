from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from models.svd import SVD
from read_data import read_data, create_tabular_data

users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
print(users.head())

ratings, movies = read_data(Path('ml-1m'))

df = create_tabular_data(ratings, users, movies)

train_set, test_set = train_test_split(df, test_size=0.2)
print(train_set.head())
# train_data, test_data, data = dp.load_dataset()

f = 20
model = SVD(train_set, f)
model.train()
model.rmse(test_set)
