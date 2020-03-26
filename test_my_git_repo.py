from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error
from DIPLOMv1.dataset import fetch_ml_ratings
from DIPLOMv1 import SVD

from sklearn.metrics import mean_absolute_error

# 20m dataset
import pandas as pd
import numpy as np
import zipfile
import urllib.request
import os

df = fetch_ml_ratings()

movies_df = pd.read_csv(
    'ml-20m/movies.csv', names=['i_id', 'title', 'genres'], sep=',', encoding='latin-1')
movies_df.drop([0], inplace=True)
movies_df['i_id'] = movies_df['i_id'].apply(pd.to_numeric)

# Create one merged DataFrame containing all the movielens data.

model = df.copy()


train = model.sample(frac=0.8)
val = model.drop(train.index.tolist()).sample(frac=0.5, random_state=8)
test = model.drop(train.index.tolist()).drop(val.index.tolist())

iterations = 100


def sample_params():
    lr = np.random.uniform(low=0.001, high=0.1,  size=1)[0]
    reg = np.random.uniform(low=0.001, high=0.1,  size=1)[0]
#     factors = np.random.randint(low = 100, high = 500,  size = 1)[0]
    factors = 64
    return lr, reg, factors


# lr, reg, factors = (0.007, 0.03, 90)
lr, reg, factors = (0.02, 0.016, 64)
svd = SVD(learning_rate=lr, regularization=reg, n_epochs=200, n_factors=factors,
          min_rating=0.5, max_rating=5)

svd.fit(X=train, X_val=val, early_stopping=True, shuffle=False)

pred = svd.predict(test)
mae = mean_absolute_error(test["rating"], pred)
rmse = np.sqrt(mean_squared_error(test["rating"], pred))
print("Test MAE:  {:.2f}".format(mae))
print("Test RMSE: {:.2f}".format(rmse))
print('{} factors, {} lr, {} reg'.format(factors, lr, reg))

# Adding our own ratings

n_m = len(model.i_id.unique())

#  Initialize my ratings
my_ratings = np.zeros(n_m)


my_ratings[4993] = 5
my_ratings[1080] = 5
my_ratings[260] = 5
my_ratings[4896] = 5
my_ratings[1196] = 5
my_ratings[1210] = 5
my_ratings[2628] = 5
my_ratings[5378] = 5

print('User ratings:')
print('-----------------')

for i, val in enumerate(my_ratings):
    if val > 0:
        print('Rated %d stars: %s' %
              (val, movies_df.loc[movies_df.i_id == i].title.values))

print("Adding your recommendations!")
items_id = [item[0] for item in np.argwhere(my_ratings > 0)]
ratings_list = my_ratings[np.where(my_ratings > 0)]
user_id = np.asarray([0] * len(ratings_list))

user_ratings = pd.DataFrame(
    list(zip(user_id, items_id, ratings_list)), columns=['u_id', 'i_id', 'rating'])

try:
    model = model.drop(columns=['timestamp'])
except:
    pass
data_with_user = model.append(user_ratings, ignore_index=True)

train_user = data_with_user.sample(frac=0.8)
val_user = data_with_user.drop(
    train_user.index.tolist()).sample(frac=0.5, random_state=8)
test_user = data_with_user.drop(
    train_user.index.tolist()).drop(val_user.index.tolist())

# lr, reg, factors = (0.007, 0.03, 90)
lr, reg, factors = (0.02, 0.016, 64)
epochs = 10  # epochs = 50

svd = SVD(learning_rate=lr, regularization=reg, n_epochs=epochs, n_factors=factors,
          min_rating=0.5, max_rating=5)

svd.fit(X=train_user, X_val=val_user, early_stopping=False,
        shuffle=False)  # early_stopping=True

pred = svd.predict(test_user)
mae = mean_absolute_error(test_user["rating"], pred)
rmse = np.sqrt(mean_squared_error(test_user["rating"], pred))
print("Test MAE:  {:.2f}".format(mae))
print("Test RMSE: {:.2f}".format(rmse))
print('{} factors, {} lr, {} reg'.format(factors, lr, reg))


def funk_svd_predict(userID, data_with_user, movies_df):
    userID = [userID]

    # all_users = data_with_user.u_id.unique()
    all_movies = data_with_user.i_id.unique()
    recommendations = pd.DataFrame(
        list(product(userID, all_movies)), columns=['u_id', 'i_id'])

    # Getting predictions for the selected userID
    pred_train = svd.predict(recommendations)
    recommendations['prediction'] = pred_train
    recommendations.head(10)

    sorted_user_predictions = recommendations.sort_values(
        by='prediction', ascending=False)

    user_ratings = data_with_user[data_with_user.u_id == userID[0]]
    user_ratings.columns = ['u_id',	'i_id', 'rating']
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = movies_df[~movies_df['i_id'].isin(user_ratings['i_id'])].\
        merge(pd.DataFrame(sorted_user_predictions).reset_index(drop=True), how='inner', left_on='i_id', right_on='i_id').\
        sort_values(by='prediction', ascending=False)  # .drop(['i_id'],axis=1)

    rated_df = movies_df[movies_df['i_id'].isin(user_ratings['i_id'])].\
        merge(pd.DataFrame(data_with_user).reset_index(drop=True),
              how='inner', left_on='i_id', right_on='i_id')
    rated_df = rated_df.loc[rated_df.u_id == userID[0]
                            ].sort_values(by='rating', ascending=False)

    return recommendations, rated_df

def printTable(table):
    display('\n'.join(table))

m = list(movies_df[movies_df["genres"] == "Horror"].i_id)
o = df[df["i_id"] == 62203].sort_values(by='rating', ascending=False)
printTable(o.iloc[0:30, :])

temp = df.groupby(['rating']).count()
printTable(temp)
# print(temp[temp["i_id"] == temp["i_id"].min()])

temp = df.groupby(['u_id']).count()
temp = temp[temp["i_id"] == temp["i_id"].min()]
printTable(temp.iloc[90:120, :])

recommendations, rated_df = funk_svd_predict(	0	, data_with_user, movies_df)

printTable(rated_df.iloc[0:20, :])

printTable(recommendations.head(30))

recommendations, rated_df = funk_svd_predict(	0	, data_with_user, movies_df)
printTable(recommendations.head(30))
