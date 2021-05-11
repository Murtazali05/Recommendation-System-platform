import numpy as np
import pandas as pd

# users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
# ratings, movies = read_data(Path('ml-1m'))
# df = create_tabular_data(ratings, users, movies)
# _, test_set = train_test_split(df, test_size=0.2)
# test_set.to_csv('unseen_set.csv')

# Loading the dataset unseen by the trained models
df = pd.read_csv('unseen_set.csv')

# dataset containing all the ground truth predictions which can be used to validate the CTR of recommendations
ratings_df = df.copy()

print(ratings_df.head())

future_df = pd.DataFrame(columns=ratings_df.columns)
present_df = pd.DataFrame(columns=ratings_df.columns)

import random
from itertools import islice

for i in range(len(ratings_df)):
    print(i)
    rated_movies = np.nonzero(np.array(ratings_df.iloc[0][1:]))[0].tolist()
    random.shuffle(rated_movies)
    num_rated = len(rated_movies)
    pn = num_rated // 2
    fn = num_rated - pn
    Inputt = iter(rated_movies)
    sets = [islice(Inputt, elem) for elem in [pn, fn]]
    present_set = [x for x in sets[0]] + [0]
    future_set = [x for x in sets[1]] + [0]

    present_set.sort()
    future_set.sort()

    pr_series = [0.0] * ratings_df.shape[1]
    fr_series = [0.0] * ratings_df.shape[1]
    for pr in present_set:
        pr_series[pr] = ratings_df.iloc[i][pr]
    pr_series = [int(pr_series[0])] + pr_series[1:]
    for fr in future_set:
        fr_series[fr] = ratings_df.iloc[i][fr]
    fr_series = [int(fr_series[0])] + fr_series[1:]

    present_df.loc[i] = pr_series
    future_df.loc[i] = fr_series

present_df['userId'] = present_df['userId'].astype(int)
future_df['userId'] = future_df['userId'].astype(int)

present_df.to_csv('present_set.csv')

future_df.to_csv('future_set.csv')
